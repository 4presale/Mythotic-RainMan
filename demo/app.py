"""Small Flask web demo for the OpenMythos library."""

import math
import time

import torch
from flask import Flask, jsonify, render_template

from open_mythos.main import OpenMythos, MythosConfig
from open_mythos.variants import (
    mythos_1b,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
    mythos_1t,
)

app = Flask(__name__)

VARIANT_FUNCS = {
    "mythos_1b": mythos_1b,
    "mythos_3b": mythos_3b,
    "mythos_10b": mythos_10b,
    "mythos_50b": mythos_50b,
    "mythos_100b": mythos_100b,
    "mythos_500b": mythos_500b,
    "mythos_1t": mythos_1t,
}

VARIANT_TABLE = [
    {"name": "mythos_1b", "dim": 2048, "experts": 64, "expert_dim": 2048, "loops": 16, "context": "4k", "output": "4k"},
    {"name": "mythos_3b", "dim": 3072, "experts": 64, "expert_dim": 4096, "loops": 16, "context": "4k", "output": "4k"},
    {"name": "mythos_10b", "dim": 4096, "experts": 128, "expert_dim": 5632, "loops": 24, "context": "8k", "output": "4k"},
    {"name": "mythos_50b", "dim": 6144, "experts": 256, "expert_dim": 9728, "loops": 32, "context": "8k", "output": "4k"},
    {"name": "mythos_100b", "dim": 8192, "experts": 256, "expert_dim": 13568, "loops": 32, "context": "1M", "output": "128k"},
    {"name": "mythos_500b", "dim": 12288, "experts": 512, "expert_dim": 23040, "loops": 48, "context": "1M", "output": "128k"},
    {"name": "mythos_1t", "dim": 16384, "experts": 512, "expert_dim": 34560, "loops": 64, "context": "1M", "output": "128k"},
]


def _small_config(attn_type: str = "mla") -> MythosConfig:
    base = dict(
        vocab_size=1000,
        dim=256,
        n_heads=8,
        max_seq_len=128,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        n_experts=8,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=64,
        lora_rank=8,
        attn_type=attn_type,
    )
    if attn_type == "gqa":
        return MythosConfig(**base, n_kv_heads=2)
    return MythosConfig(
        **base,
        n_kv_heads=8,
        kv_lora_rank=32,
        q_lora_rank=64,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )


def _fmt_params(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.1f}T"
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


@app.route("/")
def index():
    return render_template(
        "index.html",
        variants=VARIANT_TABLE,
    )


@app.route("/api/run", methods=["POST"])
def run_demo():
    """Run a tiny model forward + generate pass and return results."""
    results = {}
    torch.manual_seed(42)
    for attn_type in ("mla", "gqa"):
        cfg = _small_config(attn_type)
        model = OpenMythos(cfg)
        total = sum(p.numel() for p in model.parameters())

        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        t0 = time.time()
        logits = model(ids, n_loops=4)
        fwd_ms = (time.time() - t0) * 1000

        t0 = time.time()
        out = model.generate(ids, max_new_tokens=8, n_loops=8)
        gen_ms = (time.time() - t0) * 1000

        A = model.recurrent.injection.get_A()
        rho = A.abs().max().item()

        results[attn_type] = {
            "params": _fmt_params(total),
            "params_raw": total,
            "logits_shape": list(logits.shape),
            "gen_shape": list(out.shape),
            "fwd_ms": round(fwd_ms, 1),
            "gen_ms": round(gen_ms, 1),
            "spectral_radius": round(rho, 4),
        }
    return jsonify(results)


@app.route("/api/variant/<name>")
def variant_info(name):
    func = VARIANT_FUNCS.get(name)
    if func is None:
        return jsonify({"error": "unknown variant"}), 404
    cfg = func()
    # Estimate parameters without instantiating the full model
    embed = cfg.vocab_size * cfg.dim
    prelude_coda = 2 * (cfg.prelude_layers + cfg.coda_layers) * (
        # attention (rough: 4 projections of dim*dim)
        4 * cfg.dim * cfg.dim
        # dense FFN
        + 3 * cfg.dim * (cfg.dim * 4 // 3)
    )
    recurrent_attn = 4 * cfg.dim * cfg.dim
    recurrent_moe = 3 * cfg.dim * cfg.expert_dim * (cfg.n_experts + cfg.n_shared_experts)
    recurrent = recurrent_attn + recurrent_moe
    total = embed + prelude_coda + recurrent
    return jsonify({
        "name": name,
        "params": _fmt_params(total),
        "params_raw": total,
        "dim": cfg.dim,
        "n_heads": cfg.n_heads,
        "n_experts": cfg.n_experts,
        "expert_dim": cfg.expert_dim,
        "max_loop_iters": cfg.max_loop_iters,
        "max_seq_len": cfg.max_seq_len,
        "attn_type": cfg.attn_type,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
