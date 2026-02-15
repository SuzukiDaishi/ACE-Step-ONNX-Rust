from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

VALID_SHIFTS = [1.0, 2.0, 3.0]
VALID_TIMESTEPS = [
    1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
    0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
    0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
    0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
]
SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693, 0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3],
}


def resolve_timesteps(shift: float, timesteps: Optional[Iterable[float]], max_steps: int = 20) -> np.ndarray:
    if timesteps is not None:
        arr = [float(x) for x in timesteps]
        while arr and arr[-1] == 0.0:
            arr.pop()
        if not arr:
            arr = SHIFT_TIMESTEPS[min(VALID_SHIFTS, key=lambda x: abs(x - shift))]
        if len(arr) > max_steps:
            arr = arr[:max_steps]
        mapped = [min(VALID_TIMESTEPS, key=lambda x: abs(x - t)) for t in arr]
        return np.asarray(mapped, dtype=np.float32)

    shift_mapped = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
    return np.asarray(SHIFT_TIMESTEPS[shift_mapped], dtype=np.float32)


def ode_step(xt: np.ndarray, vt: np.ndarray, t_curr: float, t_next: float) -> np.ndarray:
    dt = np.float32(t_curr - t_next)
    return xt - vt * dt


def x0_from_noise(xt: np.ndarray, vt: np.ndarray, t_curr: float) -> np.ndarray:
    t = np.float32(t_curr)
    return xt - t * vt
