import os

from .patch import install_pytorch_patch as _install_pytorch_patch


if os.environ.get(
    "GFX1010_ATTENTION_AUTOLOAD",
    "1",
).lower() not in {"0", "false", "no", "off"}:
    _install_pytorch_patch()
