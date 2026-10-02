# AGENTS.md

## Project Overview
OpenMythos is a Python PyTorch library implementing a Recurrent-Depth Transformer (RDT) — a theoretical reconstruction of the Claude Mythos architecture. It is NOT a web application; it's a library/ML package.

## Running in Base44
- The library has no web server. A small Flask demo app (`demo/`) was added to visualize the model in the preview.
- `docker-compose.base44.yml` builds from `demo/Dockerfile` (python:3.12-slim + torch CPU + flask), bind-mounts the repo, and runs `python demo/app.py` on port 3000.
- `PYTHONPATH=/app` is required so Python can find the `open_mythos` package from the bind-mounted source.
- Flask debug mode provides live reload for edits to `demo/` files.

## Key Details
- Torch is installed from the CPU index (`https://download.pytorch.org/whl/cpu`) — no GPU needed.
- The demo runs a tiny model (dim=256, 8 experts, 4 loops) on CPU in ~100ms.
- The `/api/variant/<name>` endpoint estimates parameter counts analytically — it does NOT instantiate full-scale models (which would OOM).
- `LTIInjection.get_A()` returns a 1-D tensor (diagonal of A); spectral radius = `A.abs().max()`, NOT `torch.linalg.eigvals` (which requires 2-D input).
- No external secrets or credentials are needed.

## Testing
- `curl http://localhost:3000/` — serves the demo page
- `curl -X POST http://localhost:3000/api/run` — runs small model inference
- `curl http://localhost:3000/api/variant/mythos_3b` — returns config + estimated params
