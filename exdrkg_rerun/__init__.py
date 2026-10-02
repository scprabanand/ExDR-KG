"""ExDR-KG reviewer rerun: modular code for the revision experiments (see README.md and docs/STEP_MAP.md)."""
__version__ = "0.1.0"

import torch, argparse
if not torch.cuda.is_available() and torch.backends.mps.is_available():
    # Mock torch.cuda memory and seed methods for Mac, but keep is_available False
    # so optimizers don't try to use CUDA streams.
    torch.cuda.get_device_name = lambda x: "MPS"
    torch.cuda.manual_seed_all = lambda x: torch.mps.manual_seed(x)
    torch.cuda.reset_peak_memory_stats = lambda: None
    torch.cuda.max_memory_allocated = lambda: 0
    torch.cuda.synchronize = lambda: torch.mps.synchronize()
    # Intercept device="cuda"
    _orig_add_argument = argparse.ArgumentParser.add_argument
    def _add_argument(self, *args, **kwargs):
        if kwargs.get("default") in ("cuda", "cuda:0"):
            kwargs["default"] = "mps"
        return _orig_add_argument(self, *args, **kwargs)
    argparse.ArgumentParser.add_argument = _add_argument
    # Intercept .to("cuda")
    _orig_to = torch.Tensor.to
    def _to(self, *args, **kwargs):
        if args and args[0] in ("cuda", "cuda:0"):
            args = ("mps",) + args[1:]
        if kwargs.get("device") in ("cuda", "cuda:0"):
            kwargs["device"] = "mps"
        return _orig_to(self, *args, **kwargs)
    torch.Tensor.to = _to
    _orig_nn_to = torch.nn.Module.to
    def _nn_to(self, *args, **kwargs):
        if args and args[0] in ("cuda", "cuda:0"):
            args = ("mps",) + args[1:]
        if kwargs.get("device") in ("cuda", "cuda:0"):
            kwargs["device"] = "mps"
        return _orig_nn_to(self, *args, **kwargs)
    torch.nn.Module.to = _nn_to
