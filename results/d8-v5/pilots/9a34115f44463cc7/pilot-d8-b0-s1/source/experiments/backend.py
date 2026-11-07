"""Immutable attention kernel identity and fail-closed experiment backend checks."""

FA3_REPO = "varunneal/flash-attention-3"
# HF API verified this commit contains torch29-cxx11-cu128-x86_64-linux.
# Change this only as part of a new, consistently rerun experiment suite.
FA3_REVISION = "de87b9b5af06dd9984df595bef90b2eba44b181a"


def kernel_provenance():
    return {"repo": FA3_REPO, "revision": FA3_REVISION}


def patch_attention_loader(source):
    """Pin the historical loader; reject unexpected source instead of guessing."""
    needle = "get_kernel('varunneal/flash-attention-3').flash_attn_interface"
    if source.count(needle) != 1 or source.count("import torch\n") != 1:
        raise RuntimeError("historical attention loader no longer matches the expected source")
    source = source.replace(
        "import torch\n",
        f"import torch\n\nFA3_KERNEL_REPO = {FA3_REPO!r}\nFA3_KERNEL_REVISION = {FA3_REVISION!r}\n",
    )
    return source.replace(
        needle,
        "get_kernel(FA3_KERNEL_REPO, revision=FA3_KERNEL_REVISION).flash_attn_interface",
    )


def require_fa3(attention=None):
    """Reject silent SDPA fallback and an unpinned runtime before costly work."""
    if attention is None:
        from nanochat import flash_attention as attention
    if (getattr(attention, "FA3_KERNEL_REPO", None) != FA3_REPO
            or getattr(attention, "FA3_KERNEL_REVISION", None) != FA3_REVISION):
        raise RuntimeError("experiment requires the pinned FA3 loader; prepare the historical runtime first")
    # The pinned historical module decides dynamically; newer modules cache it.
    use_fa3 = getattr(attention, "_use_fa3", None)
    selected = use_fa3() if callable(use_fa3) else getattr(attention, "USE_FA3", False)
    if not getattr(attention, "HAS_FA3", False) or not selected:
        raise RuntimeError(
            f"experiment requires FA3 from {FA3_REPO}@{FA3_REVISION}; "
            "the loader failed or selected SDPA. Run the GPU check before training."
        )
