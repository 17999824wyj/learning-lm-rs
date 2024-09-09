use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len); // use this as a unit, the D is the lastest 2 D
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len; // get the first unit
        for i in 0..seq_len {
            let offset = base + i * total_seq_len; // in the unit, get one row's first location
            let boundary = total_seq_len - seq_len + i + 1; // WHY ?
                                                            // maybe because, use `masked` together with `softmax`, so need to calculate the `boundary`

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));
            // as for the `max`, the first step is to get num of `boundary` data from offset in data[]
            // second, it will get the `maximum` of those data

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>(); // calculate the sum

            (0..boundary).for_each(|j| data[offset + j] /= sum); // normalize
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0); // mask
        }
    }
}

// y = (w * x) / (sum{x^2} / n + epsilon).sqrt()
pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    // Ensure the same length of `n`
    let &n = y.shape().last().unwrap();
    assert!(
        n == *x.shape().last().unwrap(),
        "[ERROR] @rms_norm >> the last dimension of X & Y is not the same"
    );

    // Ensure `w` is a `1 x n` vector
    assert!(
        n == w.shape()[0],
        "[ERROR] @rms_norm >> the length of `w` is not `n`"
    );
    assert!(
        w.shape().len() == 1,
        "[ERROR] @rms_norm >> `w` is not a `1 * n` vector"
    );

    // for simplifing this func, regard X and Y as the same shape
    assert!(
        x.size() == y.size(),
        "[ERROR] @rms_norm >> NOT-SUPPORT! X & Y should have the same shape"
    );

    // used for later calculate
    let num = y.size() / n;

    // get data
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();

    // start calculate
    for i in 0..num {
        let cur_x = &_x[i * n..(i + 1) * n];
        let square_sum = cur_x.iter().fold(0f32, |acc, t| acc + t * t);
        for j in 0..n {
            _y[i * n + j] = _w[j] * cur_x[j] / ((square_sum / n as f32 + epsilon).sqrt());
        }
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(
        len == x.size(),
        "[ERROR] @silu >> x & y doesn't have the same size"
    );

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    // use sigmoid for x
    for i in 0..len {
        _y[i] *= _x[i] * sigmoid(_x[i]);
    }
}

// y = sigmoid(x) = 1 / (1 + exp(-x))
#[inline]
fn sigmoid<T>(x: T) -> f32
where
    T: std::ops::Neg<Output = T> + Copy + std::convert::From<f32> + std::convert::Into<f32>,
{
    /* Illustrations
     * Why not `pub`?
     *   -- This function is just what I do, I don't know whether teachers
     *      will provide it in the following stage.
     * Why `inline`?
     *   -- This function is based on element, so it will be used in loop
     *      or in matrix, so it's better to be `inline` to avoid function
     *      call overhead.
     * Why use Generic?
     *   -- This function should can not only get `f32` type, but also
     *      other basic types.
     */
    let exp_x = (-x).into().exp(); // Convert T to f32 to compute
    1.0 / (1.0 + exp_x)
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // ** NOT support for broadcast now! **
    // I don't know how to transporse a high-Demension matrix,
    // so just regard A & B as 2D-matrix

    // make sure 2D
    assert!(
        2 == a.shape().len(),
        "[ERROR] @matmul_transb >> A is not a 2D-matrix"
    );
    assert!(
        2 == b.shape().len(),
        "[ERROR] @matmul_transb >> B is not a 2D-matrix"
    );

    // make sure can times
    let (m, k) = (a.shape()[0], a.shape()[1]);
    assert!(
        k == b.shape()[1],
        "[ERROR] @matmul_transb >> A & B can't times, because no supported shape:\n  A -> {:?}\n  B -> {:?}",
        a.shape(), b.shape()
    );
    let n = b.shape()[0];

    // make sure the result can be added by C
    assert!(
        m == c.shape()[0],
        "[ERROR] @matmul_transb >> A & C can't add, because no supported shape:\n  A -> {:?}\n  C -> {:?}",
        a.shape(), c.shape()
    );
    assert!(
        n == c.shape()[1],
        "[ERROR] @matmul_transb >> B & C can't add, because no supported shape:\n  B -> {:?}\n  C -> {:?}",
        b.shape(), c.shape()
    );

    // get data
    let _a = a.data();
    let _b = b.data();
    let _c = unsafe { c.data_mut() };

    // C_{mn} = row-m of A * col-n of B^T
    // after B's transporse
    // C_{mn} = row-m of A * col-n of B
    for i in 0..m {
        let row_a = &_a[i * k..(i + 1) * k];
        for j in 0..n {
            let row_b = &_b[j * k..(j + 1) * k];

            let sum: f32 = row_a.iter().zip(row_b.iter()).map(|(&a, &b)| a * b).sum();
            _c[i * n + j] *= beta;
            _c[i * n + j] += alpha * sum;
        }
    }
    c.reshape(&vec![m, n]);
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_softmax() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let res = masked_softmax(&mut y);
    println!("{:?}", res);
}

#[test]
fn test_matmul_transb_broadcast_1() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![5., 6.], &vec![1, 2]);
    matmul_transb(&mut c, 1.5, &a, &b, 0.5);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![10., 11.5, 24., 25.5], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_2() {
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![5., 6.], &vec![1, 2]);
    matmul_transb(&mut c, 1.5, &a, &b, 0.5);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![5., 12., 15., 24.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_broadcast_3() {
    let mut c = Tensor::<f32>::new(vec![0., 0., 0., 0.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let b = Tensor::<f32>::new(vec![5., 6.], &vec![1, 2]);
    matmul_transb(&mut c, 1.5, &a, &b, 0.5);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![5., 12., 15., 24.], &vec![2, 2]),
        1e-3
    ));
}
