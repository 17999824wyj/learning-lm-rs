use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let view = safetensor.tensor(name).unwrap();

            // ensure that the data-type is f32
            assert!(
                view.dtype() == Dtype::F32,
                "[ERROR] @from_safetensors >> data-type should be f32, but it is {:?}",
                view.dtype()
            );

            // ensure data is 4 bytes
            let capacity = view.data().len();
            assert!(
                capacity % 4 == 0,
                "[ERROR] @from_safetensors >> f32 should be 4 bytes, but the length of 1-byte data is {:?}",
                capacity
            );

            // convert u8 -> f32
            let data = view
                .data()
                .chunks(4)
                .map(|chunk| {
                    f32::from_le_bytes(chunk.try_into().expect("Chunk is not 4 bytes long"))
                })
                .collect();
            Tensor::<f32>::new(data, &Vec::from(view.shape()))
        };

        let layer = config.num_hidden_layers;

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),

            rms_att_w: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.input_layernorm.weight")))
                .collect(),
            wq: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.self_attn.o_proj.weight")))
                .collect(),

            rms_ffn_w: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..layer)
                .map(|f| get_tensor(&format!("model.layers.{f}.mlp.down_proj.weight")))
                .collect(),

            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
