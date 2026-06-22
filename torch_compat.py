"""Force pre-2.6 ``torch.load`` behaviour for UniRig / SkinTokens checkpoints.

The Docker build injects ``import torch_compat`` as the FIRST line of UniRig's
``run.py`` (and SkinTokens' ``demo.py``), so this override is installed in
normal program flow — inside the exact process that later builds the Lightning
trainer and loads the ``.ckpt`` — before any checkpoint is read.

Why not a site-packages ``sitecustomize.py``? That relies on the interpreter
auto-importing ``sitecustomize`` during ``site`` startup, which this base
image's python does NOT do reliably (so the patch never applied and the load
kept failing). Importing from the entrypoint is deterministic.

torch>=2.6 flipped ``torch.load(weights_only=...)`` to default ``True`` and
Lightning forwards it explicitly, so the unpickler rejects UniRig's checkpoints
(they embed a python-box ``Box`` config object → ``Unsupported global:
box.box.Box``). We trust these weights, so wrap ``torch.load`` to force
``weights_only=False`` and also allowlist ``Box`` belt-and-braces.

Intentionally NOT defensive around ``import torch``: run.py/demo.py require
torch anyway, so a failure here should be loud, not silently swallowed.
"""

import torch

_orig_load = torch.load


def _load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _load

try:
    from box import Box

    torch.serialization.add_safe_globals([Box])
except Exception:
    pass
