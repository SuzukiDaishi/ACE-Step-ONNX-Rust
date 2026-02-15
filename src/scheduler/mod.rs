use ndarray::Array3;

const VALID_SHIFTS: [f32; 3] = [1.0, 2.0, 3.0];
const VALID_TIMESTEPS: [f32; 20] = [
    1.0,
    0.95454544,
    0.93333334,
    0.9,
    0.875,
    0.85714287,
    0.8333333,
    0.7692308,
    0.75,
    0.6666667,
    0.64285713,
    0.625,
    0.54545456,
    0.5,
    0.4,
    0.375,
    0.3,
    0.25,
    0.22222222,
    0.125,
];

const SHIFT_TIMESTEPS_1: [f32; 8] = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125];
const SHIFT_TIMESTEPS_2: [f32; 8] = [1.0, 0.93333334, 0.85714287, 0.7692308, 0.6666667, 0.54545456, 0.4, 0.22222222];
const SHIFT_TIMESTEPS_3: [f32; 8] = [1.0, 0.95454544, 0.9, 0.8333333, 0.75, 0.64285713, 0.5, 0.3];

pub fn resolve_timesteps(shift: f32, timesteps: Option<&[f32]>, max_steps: usize) -> Vec<f32> {
    if let Some(ts) = timesteps {
        let mut arr: Vec<f32> = ts.iter().copied().filter(|v| *v != 0.0).collect();
        if arr.is_empty() {
            arr = shift_defaults(shift).to_vec();
        }
        if arr.len() > max_steps {
            arr.truncate(max_steps);
        }
        return arr
            .iter()
            .map(|t| nearest(*t, &VALID_TIMESTEPS))
            .collect();
    }

    shift_defaults(shift).to_vec()
}

pub fn ode_step(xt: &Array3<f32>, vt: &Array3<f32>, t_curr: f32, t_next: f32) -> Array3<f32> {
    let dt = t_curr - t_next;
    let mut out = xt.clone();
    out.zip_mut_with(vt, |x, v| {
        *x -= *v * dt;
    });
    out
}

pub fn x0_from_noise(xt: &Array3<f32>, vt: &Array3<f32>, t_curr: f32) -> Array3<f32> {
    let mut out = xt.clone();
    out.zip_mut_with(vt, |x, v| {
        *x -= *v * t_curr;
    });
    out
}

fn shift_defaults(shift: f32) -> &'static [f32] {
    let mapped = nearest(shift, &VALID_SHIFTS);
    if (mapped - 1.0).abs() < 1e-6 {
        &SHIFT_TIMESTEPS_1
    } else if (mapped - 2.0).abs() < 1e-6 {
        &SHIFT_TIMESTEPS_2
    } else {
        &SHIFT_TIMESTEPS_3
    }
}

fn nearest(value: f32, candidates: &[f32]) -> f32 {
    let mut best = candidates[0];
    let mut best_dist = (value - best).abs();
    for &c in candidates.iter().skip(1) {
        let dist = (value - c).abs();
        if dist < best_dist {
            best = c;
            best_dist = dist;
        }
    }
    best
}
