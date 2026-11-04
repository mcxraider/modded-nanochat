"""Kernel pinning is verified without importing torch or allocating a GPU."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.backend import FA3_REPO, FA3_REVISION, patch_attention_loader, require_fa3
from experiments.config import BASE_COMMIT


def test_historical_loader_passes_immutable_revision_to_kernel():
    source = subprocess.check_output(
        ["git", "show", f"{BASE_COMMIT}:nanochat/flash_attention.py"],
        cwd=Path(__file__).resolve().parents[1], text=True,
    )
    tree = ast.parse(patch_attention_loader(source))
    # Execute only the loader and its identity constants, using a fake CUDA and
    # kernels module. This checks the actual call without downloading a binary.
    nodes = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name == "_load_flash_attention_3"
        or isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id.startswith("FA3_KERNEL_")
            for target in node.targets
        )
    )]
    calls = []
    interface = object()
    fake_kernel = SimpleNamespace(get_kernel=lambda *args, **kwargs: (
        calls.append((args, kwargs)) or SimpleNamespace(flash_attn_interface=interface)
    ))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, get_device_capability=lambda: (9, 0),
    ))
    with pytest.MonkeyPatch.context() as monkeypatch:
        import sys
        monkeypatch.setitem(sys.modules, "kernels", fake_kernel)
        namespace = {"torch": fake_torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<loader>", "exec"), namespace)
        assert namespace["_load_flash_attention_3"]() is interface
    assert calls == [((FA3_REPO,), {"revision": FA3_REVISION})]


def test_loader_patch_rejects_source_drift():
    with pytest.raises(RuntimeError, match="no longer matches"):
        patch_attention_loader("import torch\nget_kernel('different/kernel')")


def attention(**overrides):
    return SimpleNamespace(**{
        "FA3_KERNEL_REPO": FA3_REPO, "FA3_KERNEL_REVISION": FA3_REVISION,
        "HAS_FA3": True, "_use_fa3": lambda: True, **overrides,
    })


def test_backend_guard_accepts_pinned_fa3():
    require_fa3(attention())


@pytest.mark.parametrize("override", [
    {"HAS_FA3": False}, {"_use_fa3": lambda: False},
])
def test_backend_guard_rejects_fallback(override):
    with pytest.raises(RuntimeError, match="requires FA3"):
        require_fa3(attention(**override))


def test_backend_guard_rejects_unpinned_loader():
    with pytest.raises(RuntimeError, match="pinned FA3 loader"):
        require_fa3(attention(FA3_KERNEL_REVISION="main"))
