use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use rand::seq;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            OP::matmul_transb(
                &mut residual,
                1.,
                &hidden_states,
                &self.params.wo[layer],
                1.,
            );

            hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();

        let mut input = Tensor::new(Vec::from(token_ids), &vec![token_ids.len()]);
        let mut cache = self.new_cache();
        for _ in 0..max_len {
            let res = self.forward(&input, &mut cache);
            let token = OP::random_sample(&res, top_p, top_k, temperature);
            result.push(token);
            input = Tensor::new(vec![token], &vec![1]);
        }

        result
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // After anaylsis, the output is att_scores, input is hidden_states
    // q, k, v is `full`
    let unit_len = dqkv; // the length of one unit
    let q_group_len = n_groups * unit_len; // the length of one group of q
    let k_group_len = 1 * unit_len; // the length of one group of (k & v)

    // get datas
    let _q = q.data();
    let _k = k.data();
    let _attention_scores = unsafe { att_scores.data_mut() };

    let q_row_len = n_kv_h * n_groups * unit_len; // use this to change q_rows
    let k_row_len = n_kv_h * unit_len; // use this to change k_rows

    // 1. calculate the 'q & k'
    for m in 0..seq_len {
        for n in 0..total_seq_len {
            for i in 0..n_kv_h {
                // there are n_kv_h groups in both q, k, v

                let offset_q = m * q_row_len;
                let offset_k = n * k_row_len;

                let q_group = &_q[offset_q + i * q_group_len..offset_q + (i + 1) * q_group_len];
                let k_group = &_k[offset_k + i * k_group_len..offset_k + (i + 1) * k_group_len];

                for j in 0..n_groups {
                    // in q, one group has `n_groups` unit
                    let q_unit = &q_group[j * unit_len..(j + 1) * unit_len];
                    // in k, one group has only one unit
                    // so just need to use k_group[0..k_group_len], just k_group

                    // As for next multiplication, similar with `matmul_trans`
                    // if C = A * B^T
                    // then C[i][j] = sum(A[i][:] * B[j][:])
                    // the `*` is dot product!

                    // when calculate attentions, first locate to correct 2D table
                    // if want to locate, first need to know (seq_len * total_seq_len) is the size of each 2D table
                    // we have already finished i groups, each groups has `n_groups` members
                    // so we need to multiply `i * n_groups` to get the correct 2D table
                    // then, we are in the correct 2D table, now we need to locate to correct table
                    // so finally, we need to then multiply `j * n_groups` to get the correct table
                    let offset_attention = (seq_len * total_seq_len) * (i * n_groups + j);
                    _attention_scores[offset_attention + m * total_seq_len + n] =
                        q_unit.iter().zip(k_group).map(|(a, b)| a * b).sum::<f32>()
                            / (dqkv as f32).sqrt();
                }
            }
        }
    }

    // 2. softmax & mask `attention_scores`
    OP::masked_softmax(att_scores);

    // 3. calculate `attention_output`
    // In fact, calculate `(qk) & v`
    // now, qk is `attention_score`
    // we just need to do the similar things as above

    // get data
    let _v = v.data();
    let _hidden_states = unsafe { hidden_states.data_mut() };
    let _attention_scores = att_scores.data();

    let v_group_len = 1 * unit_len; // the same size of group (k && v)
    let qk_group_len = n_groups * (seq_len * total_seq_len); // as for qk, it is constitued by `n_kv_h` 2D tables
    let hidden_group_len = n_groups * unit_len; // the same size of group (hidden)

    let attention_row_len = total_seq_len;
    let v_row_len = n_kv_h * unit_len;
    let hidden_row_len = n_kv_h * n_groups * unit_len;

    for i in 0..n_kv_h {
        // iterate the groups

        // locate the first item of v unit 2D table
        let v_start = i * v_group_len;

        for j in 0..n_groups {
            // iterate the group member in qk's one group

            // locate the first item of attention unit 2D table
            let attention_start = i * qk_group_len + j * (seq_len * total_seq_len);

            // if iterate one row of attention 2D table, column add `total_seq_len`
            // if iterate one row of v unit 2D table, column add `n_kv_h * unit_len`

            // locate the first item of `res` unit matrix
            let hidden_start = i * hidden_group_len + j * unit_len;
            // if iterate one row of `res` unit matrix, column add `n_kv_h * n_groups * unit_len`

            for m in 0..seq_len {
                let row_vec = &_attention_scores[attention_start + m * attention_row_len
                    ..attention_start + (m + 1) * attention_row_len];

                for n in 0..unit_len {
                    let mut sum = 0.;
                    for k in 0..total_seq_len {
                        sum += row_vec[k] * _v[v_start + k * v_row_len + n];
                    }
                    _hidden_states[hidden_start + m * hidden_row_len + n] = sum;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // ensure the shape, now just support 2D
    assert!(
        2 == residual.shape().len(),
        "[ERROR] @mlp >> residual is not a 2D-matrix"
    );
    assert!(
        2 == w_gate.shape().len(),
        "[ERROR] @mlp >> w_gate is not a 2D-matrix"
    );
    assert!(
        2 == w_up.shape().len(),
        "[ERROR] @mlp >> w_up is not a 2D-matrix"
    );
    assert!(
        2 == w_down.shape().len(),
        "[ERROR] @mlp >> w_down is not a 2D-matrix"
    );

    OP::rms_norm(hidden_states, &residual, rms_w, eps);
    OP::matmul_transb(gate, 0., &hidden_states, w_gate, 1.);
    OP::matmul_transb(up, 0., &hidden_states, w_up, 1.);
    OP::silu(up, gate);
    OP::matmul_transb(residual, 1., &up, w_down, 1.);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
