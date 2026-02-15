use ndarray::ArrayD;

#[derive(Debug, Clone)]
pub struct TensorMetrics {
    pub max_abs: f64,
    pub rmse: f64,
    pub cos_sim: f64,
}

pub fn tensor_metrics(a: &ArrayD<f32>, b: &ArrayD<f32>) -> TensorMetrics {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");

    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;

    for (x, y) in a.iter().zip(b.iter()) {
        let dx = (*x as f64) - (*y as f64);
        let ad = dx.abs();
        if ad > max_abs {
            max_abs = ad;
        }
        sum_sq += dx * dx;
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }

    let n = a.len().max(1) as f64;
    let rmse = (sum_sq / n).sqrt();
    let cos_sim = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };

    TensorMetrics { max_abs, rmse, cos_sim }
}
