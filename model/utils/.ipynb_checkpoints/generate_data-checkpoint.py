import numpy as np

def synthetic_drag_coefficient(length, width, height, angle_deg, noise_std=0.01, rng=None):
    """
    Toy analytical stand-in for a real CFD result.
    Not physically accurate -- just a smooth, non-linear function of
    geometry params so the surrogate has something non-trivial to learn.
    """
    if rng is None:
        rng = np.random.default_rng()

    frontal_area = width * height
    aspect_ratio = length / (width + 1e-6)
    angle_rad = np.deg2rad(angle_deg)

    base_cd = 0.9 / (1.0 + 0.15 * aspect_ratio)          # slimmer body -> lower drag
    angle_penalty = 0.4 * np.sin(angle_rad) ** 2          # higher angle of attack -> more drag
    area_factor = 1.0 + 0.05 * (frontal_area - 1.0)       # bigger frontal area -> modestly more drag

    cd = base_cd * area_factor + angle_penalty
    cd += rng.normal(0, noise_std, size=cd.shape if hasattr(cd, "shape") else None)
    return cd

def generate_dataset(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)

    length = rng.uniform(1.0, 5.0, n_samples)
    width  = rng.uniform(0.5, 2.5, n_samples)
    height = rng.uniform(0.5, 2.5, n_samples)
    angle  = rng.uniform(0.0, 30.0, n_samples)   # degrees

    cd = synthetic_drag_coefficient(length, width, height, angle, rng=rng)

    X = np.stack([length, width, height, angle], axis=1).astype(np.float32)
    y = cd.astype(np.float32).reshape(-1, 1)
    return X, y

if __name__ == "__main__":
    X, y = generate_dataset()
    np.save("/Users/joshuahellewell/arm-project/model/data/X.npy", X)
    np.save("/Users/joshuahellewell/arm-project/model/data/y.npy", y)
    print(f"Generated {X.shape[0]} samples, X shape {X.shape}, y shape {y.shape}")
    print(f"Cd range: [{y.min():.3f}, {y.max():.3f}]")