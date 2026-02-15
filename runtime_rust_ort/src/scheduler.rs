pub const VALID_SHIFTS: [f32; 3] = [1.0, 2.0, 3.0];

pub fn shift_schedule(shift: f32) -> Vec<f32> {
    let mapped = VALID_SHIFTS
        .iter()
        .copied()
        .min_by(|a, b| (a - shift).abs().partial_cmp(&(b - shift).abs()).unwrap())
        .unwrap_or(3.0);

    match mapped as i32 {
        1 => vec![1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2 => vec![1.0, 0.93333334, 0.85714287, 0.7692308, 0.6666667, 0.54545456, 0.4, 0.22222222],
        _ => vec![1.0, 0.95454544, 0.9, 0.8333333, 0.75, 0.64285713, 0.5, 0.3],
    }
}

pub fn dt(curr: f32, next: f32) -> f32 {
    curr - next
}
