"""Shared validation helpers for the audit package.

Audit doctrine (REGLES §5): the audit runs offline, on CPU, in float64, on
checkpoints -- never on MPS, never in float32.  Training may be float32;
auditing a checkpoint therefore requires loading it into a float64 copy of
the model (e.g. ``module.double()``) before calling any audit entry point.

Enforcement is two-tier, and deliberately so:

- MODEL-facing entry points (anything taking an ``nn.Module`` or a tensor
  that will be differentiated) validate loudly via the ``require_*``
  helpers: a float32/MPS model reaching the audit means the contract was
  broken upstream, and silently casting would hide that.
- Pure array utilities (norms, products, curves) accept real inputs
  through ``as_float64_array``: float16/float32 data widens EXACTLY to
  float64 (the doctrine bans float32 COMPUTE, not float32 data), other
  real dtypes convert with standard float64 rounding, and every
  computation downstream of the conversion runs in float64. Complex inputs
  are refused on all paths through ``as_float64_array``; the single
  deliberately complex routine ``pseudospectrum_grid`` (spectral.py)
  bypasses this conversion.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def require_cpu_float64_tensor(x: Tensor, name: str) -> None:
    """Raise unless ``x`` is a float64 tensor living on CPU."""
    if not isinstance(x, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.device.type != "cpu":
        raise ValueError(
            f"{name} must live on CPU (audit is CPU-only, never MPS); "
            f"got device {x.device}"
        )
    if x.dtype != torch.float64:
        raise ValueError(
            f"{name} must be float64 (audit doctrine, REGLES §5); got {x.dtype}"
        )


def require_cpu_float64_module(module: torch.nn.Module, name: str = "predictor") -> None:
    """Raise unless every floating-point parameter/buffer is CPU float64."""
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{name} must be a torch.nn.Module, got {type(module).__name__}")
    for kind, items in (("parameter", module.named_parameters()),
                        ("buffer", module.named_buffers())):
        for pname, p in items:
            if not (p.is_floating_point() or p.is_complex()):
                continue  # integer buffers (counters, indices) are fine
            if p.is_complex() or p.device.type != "cpu" or p.dtype != torch.float64:
                raise ValueError(
                    f"{name}.{pname} ({kind}) is {p.dtype} on {p.device}; the audit "
                    "requires real CPU float64 (complex parameters are unsupported) "
                    "-- load the checkpoint into a float64 copy (e.g. "
                    "module.double()) before auditing"
                )


def as_float64_array(x: Tensor | np.ndarray) -> np.ndarray:
    """Detach/convert to a float64 numpy array (audit output convention).

    float16/float32 real inputs widen exactly to float64; other real dtypes
    convert with standard float64 rounding (see the module docstring for
    why this is not a doctrine violation). Complex inputs are refused
    loudly: numpy would otherwise cast them to float64 by silently
    DISCARDING the imaginary part (a mere ``ComplexWarning``), corrupting
    norms, products, and spectra downstream.
    """
    if isinstance(x, Tensor):
        if x.is_complex():
            raise TypeError(
                "complex tensors are unsupported on the audit's real-valued "
                "paths (silently dropping the imaginary part would corrupt "
                "the audited object)"
            )
        return x.detach().cpu().to(torch.float64).numpy()
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        raise TypeError(
            "complex arrays are unsupported on the audit's real-valued paths "
            "(silently dropping the imaginary part would corrupt the audited "
            "object)"
        )
    return arr.astype(np.float64, copy=False)
