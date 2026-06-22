"""Process-wide torch.load compatibility shim for UniRig / SkinTokens.

Python auto-imports a module named ``sitecustomize`` on interpreter startup (it
is searched for on ``sys.path`` / site-packages). We drop this into the image's
site-packages so it applies to every ``python run.py`` UniRig launches.

Why: UniRig's checkpoints are ordinary pickles that embed a ``python-box``
``Box`` config object. PyTorch 2.6 flipped the default of ``torch.load``'s
``weights_only`` argument from ``False`` to ``True``; under ``weights_only=True``
the unpickler refuses any non-tensor global (``box.box.Box``) and the checkpoint
load raises ``UnpicklingError``. We trust our own model weights, so restore the
pre-2.6 behaviour by forcing ``weights_only=False`` and, belt-and-suspenders,
allowlisting the Box global.
"""

try:
    import torch

    _orig_load = torch.load

    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = _load

    try:
        import box

        torch.serialization.add_safe_globals([box.box.Box])
    except Exception:
        pass
except Exception:
    pass
