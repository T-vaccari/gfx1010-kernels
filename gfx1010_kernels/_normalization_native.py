from functools import lru_cache


NATIVE_HIDDEN_SIZES = (128, 256, 384, 512, 768, 1024)
_THREADS = {
    128: 128,
    256: 128,
    384: 128,
    512: 128,
    768: 128,
    1024: 192,
}
_PARAMETER_GROUPS = {
    128: (80, 12, 48, 80, 96, 128),
    256: (48, 32, 12, 64, 80, 80),
    384: (8, 16, 64, 64, 80, 80),
    512: (48, 48, 48, 64, 48, 48),
    768: (32, 48, 48, 48, 48, 32),
    1024: (32, 32, 32, 24, 24, 24),
}


def _block_rows(hidden_size, row_count):
    if row_count <= 512:
        return 16
    if row_count <= 2048:
        return 16 if hidden_size in {512, 1024} else 32
    return 64


def _parameter_threads(hidden_size, row_count):
    if hidden_size == 512 and row_count == 256:
        return 256
    if hidden_size == 256 and row_count == 512:
        return 256
    if hidden_size in {768, 1024} and row_count >= 4096:
        return 256
    return 64


def _parameter_groups(hidden_size, row_count):
    if row_count <= 128:
        bucket = 0
    elif row_count <= 256:
        bucket = 1
    elif row_count <= 512:
        bucket = 2
    elif row_count <= 1024:
        bucket = 3
    elif row_count <= 2048:
        bucket = 4
    else:
        bucket = 5
    return min(row_count, _PARAMETER_GROUPS[hidden_size][bucket])


@lru_cache(maxsize=1)
def _extension():
    try:
        from . import _C
    except ImportError as error:
        return None, str(error)
    return _C, ""


def native_extension_status():
    extension, reason = _extension()
    return extension is not None, reason


def native_residual_layer_norm(
    x,
    branch,
    weight,
    bias,
    dropout_p,
    eps,
    training,
):
    extension, reason = _extension()
    if extension is None:
        raise RuntimeError(
            f"gfx1010 native extension is unavailable: {reason}"
        )
    hidden_size = x.shape[-1]
    row_count = x.numel() // hidden_size
    return tuple(extension.residual_layer_norm(
        x,
        branch,
        weight,
        bias,
        float(dropout_p),
        float(eps),
        bool(training),
        _THREADS[hidden_size],
        _block_rows(hidden_size, row_count),
        _parameter_threads(hidden_size, row_count),
        _parameter_groups(hidden_size, row_count),
    ))
