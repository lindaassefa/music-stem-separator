import numpy as np

def si_sdr(reference, estimate):
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    if reference.shape != estimate.shape or reference.size == 0:
        raise ValueError("Signals must have the same non-empty shape.")

    if reference.ndim not in (1, 2):
        raise ValueError("Expected mono or multichannel audio.")

    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Audio contains non-finite values.")

    reference = reference - reference.mean(axis=0, keepdims=True)
    estimate = estimate - estimate.mean(axis=0, keepdims=True)

    reference_energy = np.sum(reference ** 2)
    estimate_energy = np.sum(estimate ** 2)

    if reference_energy < 1e-20 or estimate_energy < 1e-20:
        return float("nan")

    scale = np.sum(estimate * reference) / reference_energy
    target = scale * reference
    error = estimate - target

    target_energy = np.sum(target ** 2)
    error_energy = np.sum(error ** 2)

    if target_energy < 1e-20:
        return float("-inf")

    if error_energy < 1e-20:
        return float("inf")

    return float(10 * np.log10(target_energy / error_energy))
