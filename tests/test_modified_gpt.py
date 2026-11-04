"""Behavior checks for the historical experiment model (independent of exploratory GPT)."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import types

import pytest
import torch

from nanochat.modified_gpt import GPT, GPTConfig, apply_rotary_emb
from nanochat.engine import KVCache

ROOT = Path(__file__).resolve().parents[1]


def tiny(kv=4, fraction=1.0, ve=True):
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=3, n_head=4, n_kv_head=kv, n_embd=64, rotary_fraction=fraction,
                       value_embeds=ve)
    model = GPT(config)
    model.init_weights()
    # Exercise attention: historical initialization zeros the output projections.
    for block in model.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight, std=0.05)
        torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.05)
    return model


def test_full_rope_matches_historical_outputs_loss_and_gradients():
    source = subprocess.check_output(["git", "show", "324e69c:nanochat/gpt.py"], cwd=ROOT, text=True)
    # This checkout has the newer optimizer; no optimizer is needed for forward/backward equivalence.
    source = source.replace("from nanochat.optim import MuonAdamW, DistMuonAdamW", "")
    historical = types.ModuleType("historical_gpt")
    exec(compile(source, "historical_gpt.py", "exec"), historical.__dict__)
    torch.manual_seed(17)
    model = tiny()
    config = asdict(model.config)
    config.pop("rotary_fraction")
    config.pop("mlp_ratio")
    config.pop("value_embeds")
    ref = historical.GPT(historical.GPTConfig(**config))
    ref.init_weights()
    ref.load_state_dict(model.state_dict())
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    torch.testing.assert_close(model.cos, ref.cos, rtol=0, atol=0)
    torch.testing.assert_close(model(ids), ref(ids), rtol=0, atol=0)
    loss, ref_loss = model(ids, targets=ids), ref(ids, targets=ids)
    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    loss.backward()
    ref_loss.backward()
    for (_, parameter), (_, reference) in zip(model.named_parameters(), ref.named_parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


def test_partial_rotary_changes_prefix_and_preserves_suffix():
    x = torch.randn(2, 8, 4, 128)
    angles = torch.randn(1, 8, 1, 48)
    result = apply_rotary_emb(x, angles.cos(), angles.sin())
    torch.testing.assert_close(result[..., 96:], x[..., 96:], rtol=0, atol=0)
    assert not torch.equal(result[..., :96], x[..., :96])
    torch.testing.assert_close(result.norm(dim=-1), x.norm(dim=-1))


@pytest.mark.parametrize("kv,fraction", [(4, 1.0), (2, 1.0), (2, 0.75)])
def test_last_token_logits_match_full_prefill_and_update_entire_cache(kv, fraction):
    torch.manual_seed(23)
    model = tiny(kv, fraction).eval()
    ids = torch.randint(0, model.config.vocab_size, (2, 16))
    cache = KVCache(2, kv, 32, 16, model.config.n_layer, "cpu", torch.float32)
    projected_shapes = []
    with torch.inference_mode():
        full = model(ids)
        hook = model.lm_head.register_forward_pre_hook(
            lambda module, args: projected_shapes.append(tuple(args[0].shape)))
        try:
            prefill = model(ids[:, :11], kv_cache=cache, last_token_only=True)
            assert cache.get_pos() == 11
            decode = model(ids[:, 11:12], kv_cache=cache, last_token_only=True)
            uncached = model(ids, last_token_only=True)
        finally:
            hook.remove()
    # Check the head's input, not merely the returned slice: this prevents B x T x V allocation.
    assert projected_shapes == [(2, 1, 64)] * 3
    assert prefill.shape == decode.shape == uncached.shape == (2, 1, 64)
    assert cache.get_pos() == 12
    torch.testing.assert_close(prefill, full[:, 10:11], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(decode, full[:, 11:12], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(uncached, full[:, -1:], rtol=1e-4, atol=1e-5)


def test_last_token_logits_reject_loss_targets():
    model = tiny()
    ids = torch.zeros((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="targets"):
        model(ids, targets=ids, last_token_only=True)


@pytest.mark.parametrize("kv,fraction,ve", [(4, 1.0, True), (2, 1.0, True), (2, 0.75, True), (4, 1.0, False), (2, 1.0, False)])
def test_cached_decoding_and_projection_shapes(kv, fraction, ve):
    torch.manual_seed(23)
    model = tiny(kv, fraction, ve).eval()
    cfg = model.config
    cache = KVCache(2, kv, 32, 16, cfg.n_layer, "cpu", torch.float32)
    assert cache.k_cache.shape == (3, 2, 32, kv, 16)
    assert model.transformer.h[0].attn.c_k.weight.shape == (kv * 16, 64)
    if ve:
        assert model.value_embeds["0"].weight.shape == (64, kv * 16)
    assert model.cos.shape[-1] == cfg.rotary_dim // 2
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.inference_mode():
        full = model(ids)
        # Chunked prefill and multiple sequential decodes cross the short attention window.
        parts = [model(ids[:, :5], kv_cache=cache), model(ids[:, 5:11], kv_cache=cache)]
        parts += [model(ids[:, i:i+1], kv_cache=cache) for i in range(11, 16)]
    torch.testing.assert_close(torch.cat(parts, dim=1), full, rtol=1e-4, atol=1e-5)
    assert cache.get_pos() == 16


@pytest.mark.parametrize("kwargs", [
    {"rotary_fraction": 0}, {"rotary_fraction": 1.1}, {"rotary_fraction": float("nan")},
    {"rotary_fraction": 0.7}, {"rotary_fraction": True}, {"n_kv_head": 0},
    {"n_head": 5}, {"n_kv_head": 4}, {"window_pattern": ""}, {"n_layer": 0},
    {"mlp_ratio": 0}, {"mlp_ratio": float("nan")}, {"mlp_ratio": 4.3}, {"mlp_ratio": -4.0},
    {"value_embeds": "yes"}, {"value_embeds": 1},
])
def test_invalid_model_config(kwargs):
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


@pytest.mark.parametrize("ve", [True, False])
def test_checkpoint_dispatch_and_generation(tmp_path, monkeypatch, ve):
    from nanochat import checkpoint_manager
    model = tiny(2, 0.75, ve).eval()
    config = asdict(model.config)
    checkpoint_manager.save_checkpoint(str(tmp_path), 1, model.state_dict(), None,
        {"model_type": "modified_gpt_324e69c", "model_config": config, "step": 1})
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: types.SimpleNamespace(get_vocab_size=lambda: 64))
    loaded, _, _ = checkpoint_manager.build_model(str(tmp_path), 1, torch.device("cpu"), "eval")
    assert isinstance(loaded, GPT)
    assert asdict(loaded.config) == config
    ids = torch.randint(0, 64, (1, 8))
    torch.testing.assert_close(loaded(ids), model(ids), rtol=0, atol=0)
    assert list(loaded.generate(ids[0].tolist(), 3, temperature=0)) == list(model.generate(ids[0].tolist(), 3, temperature=0))


def test_legacy_checkpoint_dispatch_is_preserved(tmp_path, monkeypatch):
    from nanochat.gpt import GPT as OriginalGPT, GPTConfig as OriginalConfig
    from nanochat import checkpoint_manager
    config = OriginalConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
    model = OriginalGPT(config)
    model.init_weights()
    checkpoint_manager.save_checkpoint(str(tmp_path), 1, model.state_dict(), None, {"model_config": asdict(config), "step": 1})
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: types.SimpleNamespace(get_vocab_size=lambda: 64))
    loaded, _, _ = checkpoint_manager.build_model(str(tmp_path), 1, torch.device("cpu"), "eval")
    assert isinstance(loaded, OriginalGPT)


def test_mlp_ratio_default_is_historical_and_widening_adds_expected_parameters():
    base = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64)
    wide = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, n_embd=64, mlp_ratio=6.0)
    assert base.mlp_hidden == 256 and wide.mlp_hidden == 384
    count = lambda c: sum(p.numel() for p in GPT(c).transformer.h.parameters())
    assert count(wide) - count(base) == 2 * (2 * 64 * 128)  # 2 layers x (c_fc + c_proj) x 64 x 128 extra hidden units


def test_pm_ratio_recovers_removed_kv_parameters_at_d8_and_d12():
    """E3 premise: widening the MLP to 4.5x offsets the scaling parameters GQA-2 removes (to within ve_gate)."""
    from experiments.config import DEPTHS
    for depth in (8, 12):
        layers, embd, heads = DEPTHS[depth]
        with torch.device("meta"):
            cfg = lambda kv, ratio: GPTConfig(n_layer=layers, n_head=heads, n_kv_head=kv, n_embd=embd, mlp_ratio=ratio)
            scaling = lambda c: (lambda n: n["transformer_matrices"] + n["lm_head"])(GPT(c).num_scaling_params())
            mha, gqa, pm = scaling(cfg(heads, 4.0)), scaling(cfg(heads // 2, 4.0)), scaling(cfg(heads // 2, 4.5))
        assert gqa < mha
        assert abs(pm - mha) < 0.0001 * mha  # only the ve_gate remainder is left


def test_value_embeds_off_removes_exactly_the_embeddings_and_gates(monkeypatch):
    # The pinned runtime's optimizer is not in this checkout; only the parameter grouping is under test.
    import sys
    record = lambda groups: types.SimpleNamespace(param_groups=groups)
    monkeypatch.setitem(sys.modules, "nanochat.optim", types.SimpleNamespace(MuonAdamW=record, DistMuonAdamW=record))
    on, off = tiny(2), tiny(2, ve=False)
    assert len(off.value_embeds) == 0 and all(block.attn.ve_gate is None for block in off.transformer.h)
    removed = sum(p.numel() for p in on.value_embeds.parameters()) + sum(
        block.attn.ve_gate.weight.numel() for block in on.transformer.h if block.attn.ve_gate is not None)
    assert sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters()) == removed
    assert on.estimate_flops() - off.estimate_flops() == 6 * (removed - sum(p.numel() for p in on.value_embeds.parameters()))
    # VE off: no empty optimizer group. It loses exactly the VE AdamW group and the ve_gate Muon shape group.
    kinds = lambda m: sorted((g["kind"], tuple(sorted({tuple(p.shape) for p in g["params"]}))) for g in m.setup_optimizer().param_groups)
    assert all(g["params"] for g in off.setup_optimizer().param_groups)
    assert sorted(set(kinds(on)) - set(kinds(off))) == [("adamw", ((64, 32),)), ("muon", ((2, 32),))]


def test_benchmark_kv_bytes_match_allocated_cache_and_scale_with_kv_heads():
    from experiments.benchmark import kv_cache_bytes_allocated, kv_cache_bytes_analytic
    sizes = {}
    for kv in (4, 2, 1):
        cache = KVCache(3, kv, 20, 16, 5, torch.device("cpu"), torch.bfloat16)
        analytic = kv_cache_bytes_analytic(5, 3, 20, kv, 16)
        assert kv_cache_bytes_allocated(cache) == analytic
        sizes[kv] = analytic
    assert sizes[4] == 2 * sizes[2] == 4 * sizes[1]
