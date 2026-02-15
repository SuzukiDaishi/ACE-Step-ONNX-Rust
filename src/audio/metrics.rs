use ndarray::Array2;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct AudioMetrics {
    pub rmse: f64,
    pub max_abs: f64,
    pub snr_db: f64,
    pub corr: f64,
}

pub fn audio_metrics(a: &Array2<f32>, b: &Array2<f32>) -> AudioMetrics {
    let channels = a.shape()[0].min(b.shape()[0]);
    let length = a.shape()[1].min(b.shape()[1]);
    let mut max_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut sig = 0.0f64;

    for c in 0..channels {
        for t in 0..length {
            let xa = a[(c, t)] as f64;
            let xb = b[(c, t)] as f64;
            let diff = xa - xb;
            let ad = diff.abs();
            if ad > max_abs {
                max_abs = ad;
            }
            sum_sq += diff * diff;
            dot += xa * xb;
            na += xa * xa;
            nb += xb * xb;
            sig += xa * xa;
        }
    }

    let n = (channels * length).max(1) as f64;
    let rmse = (sum_sq / n).sqrt();
    let noise = sum_sq / n;
    let signal = sig / n;
    let snr_db = if noise > 0.0 {
        10.0 * (signal / noise).log10()
    } else {
        f64::INFINITY
    };
    let corr = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };

    AudioMetrics {
        rmse,
        max_abs,
        snr_db,
        corr,
    }
}
