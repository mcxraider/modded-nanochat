"""
Tests for the modded GPT variant (partial RoPE, GQA, SSL window pattern).

Run: python -m pytest tests/test_modded_gpt.py -v -s
"""
import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.modded_gpt import GPT, ModGPTConfig, apply_rotary_embed


def tiny_model():
    cfg = ModGPTConfig(sequence_len=128, vocab_size=256, n_layer=4)
    model = GPT(cfg)
    model.init_weights()
    return cfg, model


def test_n_kv_head_follows_n_head():
    # n_kv_head is derived, so bumping n_head must move it too
    assert ModGPTConfig(n_head=12).n_kv_head == 6
    assert ModGPTConfig(n_head=6, query_kv_ratio=3).n_kv_head == 2


def test_rotary_is_a_rotation():
    # zero angle is identity; the rotation preserves norm
    x = torch.randn(1, 4, 6, 8)
    d = x.shape[-1] // 2
    ones, zeros = torch.ones(1, 4, 1, d), torch.zeros(1, 4, 1, d)
    torch.testing.assert_close(apply_rotary_embed(x, ones, zeros), x)
    theta = torch.rand(1, 4, 1, d)
    y = apply_rotary_embed(x, theta.cos(), theta.sin())
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))


def test_window_pattern_ssl():
    cfg, model = tiny_model()
    short = -(-cfg.sequence_len // 4 // 128) * 128
    lefts = [w for w, _ in model.window_sizes]
    assert lefts == [short, short, cfg.sequence_len, cfg.sequence_len]  # last layer forced long


def test_train_forward_and_backward():
    torch.manual_seed(0)
    _, model = tiny_model()
    idx = torch.randint(0, 256, (2, 32))
    loss = model(idx, targets=idx)
    assert loss.isfinite()
    loss.backward()
    # value-residual path must not be in-place: c_v has to receive gradient
    assert model.transformer.h[0].attn.c_v.weight.grad is not None


def test_kvcache_decode_matches_full_forward():
    torch.manual_seed(0)
    cfg, model = tiny_model()
    model.eval()
    head_dim = cfg.n_embd // cfg.n_head
    kv = KVCache(1, cfg.n_kv_head, cfg.sequence_len, head_dim, cfg.n_layer, "cpu", COMPUTE_DTYPE)
    idx = torch.randint(0, 256, (1, 16))
    with torch.inference_mode():
        ref = model(idx)[:, -1, :]
        model(idx[:, :-1], kv_cache=kv)  # prefill
        got = model(idx[:, -1:], kv_cache=kv)[:, -1, :]
    torch.testing.assert_close(ref, got, rtol=1e-4, atol=1e-4)
