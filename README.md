# AI Weight Painting — RunPod Serverless worker (UniRig / SkinTokens)

A **separate, admin-only, GPU** weight-painting backend that runs a learned
skinning model to predict per-vertex skin weights for our **existing** skeleton.
It is fully isolated from the default Blender bone-heat worker
(`runpod/blender-weights`) — different endpoint, different env var, different
B2 key prefix.

## How it fits together

```
RigUnified (admin toggle: Blender | UniRig | SkinTokens)
  └─ requestAiAutoweight()  src/lib/rigging/autoweight-ai-runpod.ts
       └─ POST /api/autoweight-ai-runpod/run   (admin Bearer token, gated)
            └─ RunPod  https://api.runpod.ai/v2/$RUNPOD_AIWEIGHT_ENDPOINT_ID/run
                 └─ this worker (handler.py)
```

The worker speaks the **same B2 contract** as the blender-weights worker:
`input_url` (gzipped SMW1 mesh) in, `output_put_url` (gzipped weights JSON) out.

## Pipeline (handler.py)

1. Decode SMW1 → `{vertices (N,3), triangles (F,3), bones[{name,head,tail,parent}]}`.
2. `blender_build_input.py` (bpy) builds a GLB = mesh + armature with **our**
   bone names.
3. Run the chosen model's official skinning inference, existing-skeleton mode:
   - **unirig** → `bash launch/inference/generate_skin.sh --input in.glb --output skin.fbx`
   - **skintokens** → `python demo.py --input in.glb --output skin.glb --use_skeleton --use_transfer`
4. `blender_read_skin.py` (bpy) reads the skinned result → per-vertex vertex-group weights.
5. NN-transfer (KDTree, in unit-normalized space so it's robust to UniRig's
   internal rescale/decimation) onto our **original full-res** vertices; prune,
   keep top-4, renormalize.
6. Emit `{ weights, weight_method: 'AI_UNIRIG'|'AI_SKINTOKENS', diagnostics, elapsed }`.

The client treats `AI_*` methods as already-smooth (skips the bone-heat cleanup
tower) and applies the weights exactly like the Blender ones.

## Files

| File | Role |
|------|------|
| `Dockerfile` | CUDA 12.1 + py3.11 + torch 2.3.1 + spconv/scatter/cluster/flash-attn + bpy 4.2; clones UniRig + SkinTokens; pre-downloads the UniRig skin ckpt |
| `handler.py` | RunPod serverless entry; B2 I/O; orchestration; weight transfer |
| `blender_build_input.py` | JSON mesh+skeleton → GLB (mesh + named armature) |
| `blender_read_skin.py` | skinned FBX/GLB → per-vertex group weights JSON |

## Deploy (GitHub-repo-backed RunPod build — same model as blender-weights)

1. Create a public GitHub repo, e.g. `romesvonwolf/ai-weights-serverless`, with
   `Dockerfile`, `handler.py`, `blender_build_input.py`, `blender_read_skin.py`
   at the repo root (copy from this folder).
2. RunPod console → **Serverless → New Endpoint → Import from GitHub** → select
   the repo, branch `main`. RunPod builds the image (tag = git short sha).
3. Endpoint config:
   - **GPU**: 24 GB class (L4 / A5000 / 4090 / L40S). UniRig needs ~8 GB; leave headroom.
   - **Container disk**: ≥ 30 GB (CUDA + torch + bpy + checkpoints are large).
   - Min workers **0**, Max **1–2**, FlashBoot **on**, execution timeout **1800s**.
4. Copy the endpoint id into `.env.local`:
   ```
   RUNPOD_AIWEIGHT_ENDPOINT_ID=<id>
   ```
   Restart the dev server (it reads `.env.local` at startup). `RUNPOD_API_KEY`
   and the `B2_*` vars are already shared.

To "redeploy", push to `main` (RunPod rebuilds). To "rollback", revert the commit.

## Test (live)

```
node scripts/weight-lab/run_aiweights.cjs scripts/weight-lab/payload.json scripts/weight-lab/weights_ai.json unirig
node scripts/weight-lab/run_aiweights.cjs scripts/weight-lab/payload.json scripts/weight-lab/weights_skintokens.json skintokens
```

Then visualize with the existing weight-lab tools (`render_one.py`,
`region_audit.py`, `pony_hand_check.py`).

## Notes / known risks (iterate live)

- **First build** is GPU/CUDA-dependent; `flash-attn` and `spconv` wheels are the
  usual failure points — adjust the pinned versions in the Dockerfile if the
  RunPod build log complains.
- **Orientation**: UniRig is trained Y-up (Blender). The input GLB is exported
  `export_yup=True`. If results look rotated, revisit axis handling in
  `blender_build_input.py`.
- **SkinTokens** is bleeding-edge; if its repo layout/flags differ from
  `demo.py --use_skeleton`, the `unirig` path still works. Confirm against the
  current SkinTokens README and adjust `_run_skintokens` in `handler.py`.
- **Checkpoint auth**: if HF gates the checkpoint, add an `HF_TOKEN` env var to
  the endpoint and `huggingface-cli login` in the Dockerfile.
