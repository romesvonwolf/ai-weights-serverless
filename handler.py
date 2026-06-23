"""
RunPod Serverless Handler — AI weight painting (UniRig / SkinTokens).

A SEPARATE GPU worker from the Blender bone-heat one (runpod/blender-weights).
It runs a learned skinning model that predicts per-vertex skin weights for our
EXISTING skeleton, then maps them back onto our exact full-res mesh.

I/O contract is identical to the blender-weights worker so the Next.js routes
and B2 transport are reused verbatim:
  INPUT  : `input_url` → gzipped 'SMW1' binary {vertices,triangles,bones}.
  OUTPUT : gzip the weights JSON and PUT it to the presigned `output_put_url`.
  We return only a small summary.

Job input:
  {
    "input_url": "https://.../input.smw.gz",
    "output_put_url": "https://...?X-Amz-...",
    "output_content_type": "application/octet-stream",
    "timeout": 1800,
    "model": "unirig" | "skintokens"     # default "unirig"
  }

Pipeline (per job):
  1. decode SMW1 → {vertices (N,3), triangles (F,3), bones[{name,head,tail,parent}]}
  2. Blender (bpy, subprocess) builds a GLB = mesh + armature with OUR bone names.
  3. run the model's official skinning inference (existing-skeleton mode):
       unirig     → bash launch/inference/generate_skin.sh --input in.glb --output skin.fbx
       skintokens → python demo.py --input in.glb --output skin.glb --use_skeleton --use_transfer
  4. Blender reads the skinned result → per-vertex vertex-group weights + verts.
  5. NN-transfer (normalized space) those weights onto our ORIGINAL full-res
     vertices, prune + top-4 + renormalize.
  6. emit { weights, weight_method, diagnostics, elapsed } → B2.
"""

import os
import sys
import json
import time
import gzip
import base64
import struct
import itertools
import subprocess
import tempfile
import traceback
import urllib.request
from array import array

import numpy as np
from scipy.spatial import cKDTree
import runpod

UNIRIG_DIR = os.environ.get("UNIRIG_DIR", "/opt/UniRig")
SKINTOKENS_DIR = os.environ.get("SKINTOKENS_DIR", "/opt/SkinTokens")
PYBIN = os.environ.get("PYBIN", sys.executable)
DEFAULT_TIMEOUT = int(os.environ.get("AIWEIGHT_TIMEOUT", "1800"))
FACES_TARGET = int(os.environ.get("AIWEIGHT_FACES_TARGET", "50000"))
MAX_INLINE_OUTPUT = int(os.environ.get("MAX_INLINE_OUTPUT", str(6 * 1024 * 1024)))
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# B2 transport (identical contract to runpod/blender-weights/handler.py)
# --------------------------------------------------------------------------- #
def _download(url, timeout=300):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-weights-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _maybe_gunzip(raw):
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw


def _decode_smw1(buf):
    if len(buf) < 16 or buf[0:4] != b"SMW1":
        raise ValueError("not an SMW1 blob")
    Vn, Tn, Bn = struct.unpack_from("<III", buf, 4)
    off = 16
    vbytes = Vn * 3 * 4
    verts_arr = array("f")
    verts_arr.frombytes(bytes(buf[off:off + vbytes]))
    off += vbytes
    if sys.byteorder != "little":
        verts_arr.byteswap()
    tbytes = Tn * 3 * 4
    tris_arr = array("i")
    tris_arr.frombytes(bytes(buf[off:off + tbytes]))
    off += tbytes
    if sys.byteorder != "little":
        tris_arr.byteswap()
    bones = json.loads(bytes(buf[off:off + Bn]).decode("utf-8"))
    vertices = np.array(verts_arr, dtype=np.float32).reshape(-1, 3)
    triangles = np.array(tris_arr, dtype=np.int64).reshape(-1, 3)
    return {"vertices": vertices, "triangles": triangles, "bones": bones}


def _parse_mesh_bytes(raw):
    if len(raw) >= 4 and raw[0:4] == b"SMW1":
        return _decode_smw1(raw)
    d = json.loads(raw)
    return {
        "vertices": np.asarray(d["vertices"], dtype=np.float32).reshape(-1, 3),
        "triangles": np.asarray(d["triangles"], dtype=np.int64).reshape(-1, 3),
        "bones": d["bones"],
    }


def _put(url, data, content_type, timeout=300):
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _load_input(ji):
    if ji.get("input_url"):
        return _parse_mesh_bytes(_maybe_gunzip(_download(ji["input_url"])))
    if ji.get("mesh_gzip_b64"):
        return _parse_mesh_bytes(gzip.decompress(base64.b64decode(ji["mesh_gzip_b64"])))
    return {
        "vertices": np.asarray(ji["vertices"], dtype=np.float32).reshape(-1, 3),
        "triangles": np.asarray(ji["triangles"], dtype=np.int64).reshape(-1, 3),
        "bones": ji["bones"],
    }


# --------------------------------------------------------------------------- #
# Weight transfer helpers
# --------------------------------------------------------------------------- #
def _normalize_uniform(pts):
    """Center to bbox center and scale by the MAX half-extent (uniform), so the
    cloud fits in [-1,1] with proportions PRESERVED. This is exactly UniRig's
    own normalization (merge.denormalize_vertices: scale = max(extent)/2). We use
    it for the discrete orientation search, where preserving proportions is what
    makes the correct axis frame win unambiguously."""
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) * 0.5
    scale = float((hi - lo).max()) * 0.5
    if scale < 1e-9:
        scale = 1.0
    return (pts - center) / scale


def _best_orientation(src_verts, dst_verts, num=16384):
    """Find the axis permutation + sign flip that best aligns src to dst by
    minimizing mean nearest-neighbor distance, in UNIFORM-normalized (proportion-
    preserving) space. This mirrors UniRig's merge.get_correct_orientation_kdtree.
    Doing it in uniform (not per-axis) space is essential: in a per-axis unit cube
    every axis is the same length, so permutations match spuriously and a wrong
    frame can win; with proportions preserved, the true frame is unambiguous.

    Returns (perm, signs, diagnostics). The caller applies perm/signs to the RAW
    src verts before the per-axis NN match.
    """
    su = _normalize_uniform(src_verts.astype(np.float64))
    du = _normalize_uniform(dst_verts.astype(np.float64))
    rng = np.random.default_rng(0)
    a = su if len(su) <= num else su[rng.permutation(len(su))[:num]]
    b = du if len(du) <= num else du[rng.permutation(len(du))[:num]]
    perms = list(itertools.permutations((0, 1, 2)))
    signs_list = [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)]
    best = None
    ident_loss = None
    for perm in perms:
        pa = a[:, perm]
        for signs in signs_list:
            t = pa * np.asarray(signs, dtype=pa.dtype)
            d, _ = cKDTree(t).query(b)
            loss = float(d.mean())
            if perm == (0, 1, 2) and signs == (1, 1, 1):
                ident_loss = loss
            if best is None or loss < best[0]:
                best = (loss, perm, signs)
    _, perm, signs = best
    # Gate: only adopt a non-identity frame if it's CLEARLY better (>5%). We
    # control both ends of the glTF/FBX round-trip, so the frame is usually
    # already identity; a near-tie "win" is almost always a spurious mirror of a
    # roughly-symmetric body (e.g. a front/back Z-flip) that scrambles limbs.
    flipped = (perm != (0, 1, 2)) or (tuple(signs) != (1, 1, 1))
    if flipped and ident_loss is not None and best[0] > ident_loss * 0.95:
        perm, signs = (0, 1, 2), (1, 1, 1)
        chosen_loss = ident_loss
    else:
        chosen_loss = best[0]
    return perm, signs, {
        "perm": list(perm),
        "signs": list(signs),
        "nn_loss": round(chosen_loss, 5),
        "nn_loss_best": round(best[0], 5),
        "nn_loss_identity": round(ident_loss, 5) if ident_loss is not None else None,
    }


def _anchor_points(src_joints, dst_joints):
    """Shared-skeleton correspondences: bone head+tail positions present in both
    the source (model output) frame and our input frame, as (src_pts, dst_pts)."""
    sp, dp = [], []
    for nm, sj in src_joints.items():
        dj = dst_joints.get(nm)
        if not dj:
            continue
        for key in ("head", "tail"):
            if key in sj and key in dj:
                sp.append(sj[key])
                dp.append(dj[key])
    if not sp:
        return None, None
    return np.asarray(sp, dtype=np.float64), np.asarray(dp, dtype=np.float64)


def _fit_peraxis_affine(src_pts, dst_pts):
    """Per-axis least-squares affine dst ≈ a*src + b (a,b are (3,)). Diagonal
    (no rotation) because, after the discrete frame is fixed, source and target
    differ only by UniRig's per-axis normalize (a center + per-axis scale), which
    this exactly inverts. Solved independently per axis."""
    a = np.ones(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for ax in range(3):
        s = src_pts[:, ax]
        d = dst_pts[:, ax]
        sm = s.mean()
        dm = d.mean()
        var = float(((s - sm) ** 2).sum())
        if var < 1e-12:
            a[ax] = 1.0
            b[ax] = dm - sm
        else:
            a[ax] = float(((s - sm) * (d - dm)).sum()) / var
            b[ax] = dm - a[ax] * sm
    return a, b


def _transfer_weights(src_verts, src_weights, dst_verts, src_joints, dst_joints, k=6, alpha=8.0):
    """Transfer weights from src (UniRig's predicted skin on its sampled proxy)
    to dst (our full-res mesh).

    The proxy is UniRig's input mesh after a per-axis `normalize_into [-1,1]`, so
    it is a NON-uniformly scaled (inflated) version of our mesh — a plain
    normalized NN match then drags torso verts onto limb proxy verts (legs were
    the worst). But src and dst share the SAME skeleton, so we register them with
    the bones as anchors: pick the discrete frame (uniform-space search), then fit
    a per-axis affine from the shared bone head/tail positions that maps the proxy
    back onto our mesh, EXACTLY inverting UniRig's normalize. The distance-weighted
    NN blend then runs in our own coordinate frame.

    src_weights: (Ns, J) dense. Returns (Nd, J) dense and an orientation diag.
    """
    perm, signs, orient = _best_orientation(src_verts, dst_verts)
    se = src_verts.max(axis=0) - src_verts.min(axis=0)
    de = dst_verts.max(axis=0) - dst_verts.min(axis=0)
    orient["src_extent"] = [round(float(x), 3) for x in se]
    orient["dst_extent"] = [round(float(x), 3) for x in de]
    signs = np.asarray(signs, dtype=np.float64)

    src = src_verts.astype(np.float64)[:, perm] * signs
    dst = dst_verts.astype(np.float64)
    sp, dp = _anchor_points(src_joints, dst_joints)
    if sp is not None and len(sp) >= 3:
        spo = sp[:, perm] * signs
        a, b = _fit_peraxis_affine(spo, dp)
        src = src * a + b
        orient["affine_scale"] = [round(float(x), 4) for x in a]
        # residual: how well the anchors line up after the fit (sanity)
        res = np.linalg.norm((spo * a + b) - dp, axis=1).mean()
        diag = float(np.linalg.norm(dst.max(0) - dst.min(0))) or 1.0
        orient["anchor_residual"] = round(float(res / diag), 4)
        orient["anchors"] = int(len(sp))
        space = "ourframe"
    else:
        # no usable skeleton anchors: fall back to uniform-normalized match
        src = _normalize_uniform(src)
        dst = _normalize_uniform(dst)
        space = "uniform"
    orient["transfer_space"] = space

    # distance weighting in a scale-consistent space (normalize both by dst diag)
    diag = float(np.linalg.norm(dst.max(0) - dst.min(0))) or 1.0
    k = min(k, len(src))
    tree = cKDTree(src)
    dist, idx = tree.query(dst, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    w = np.exp(-alpha * dist / diag)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum < 1e-12] = 1e-12
    out = (src_weights[idx] * w[..., None]).sum(axis=1) / w_sum
    return out, orient


def _dominant_quality(verts, dense, names, joints):
    """Skinning quality in a mesh's OWN frame: assign each vertex to its
    dominant (argmax) bone, then measure how far each bone's dominant-vertex
    centroid sits from that bone's midpoint, normalized by the bbox diagonal.
    Used on UniRig's RAW output to separate prediction quality from our transfer.
    """
    if dense.shape[0] == 0 or not joints:
        return {}
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo)) or 1.0
    dom = dense.argmax(axis=1)  # (Ns,)
    rows = []
    for j, nm in enumerate(names):
        if nm not in joints:
            continue
        sel = verts[dom == j]
        if len(sel) == 0:
            continue
        cen = sel.mean(axis=0)
        jm = joints[nm]
        mid = (np.asarray(jm["head"], dtype=np.float64) + np.asarray(jm["tail"], dtype=np.float64)) * 0.5
        rel = float(np.linalg.norm(cen - mid) / diag)
        rows.append((rel, nm, int(len(sel))))
    if not rows:
        return {}
    rows.sort(reverse=True)
    mean_rel = round(sum(r[0] for r in rows) / len(rows), 4)
    well = sum(1 for r in rows if r[0] < 0.12)
    return {
        "mean_rel": mean_rel,
        "well_placed": f"{well}/{len(rows)}",
        "worst": [{"bone": r[1], "rel": round(r[0], 3), "n": r[2]} for r in rows[:6]],
    }


def _densify_topk_normalize(dense, names, prune=0.02, top_k=4):
    """dense (N, J) → sparse {boneName: {vidx: w}} keeping top-k per vertex."""
    N, J = dense.shape
    weights = {nm: {} for nm in names}
    for v in range(N):
        row = dense[v]
        if top_k < J:
            keep = np.argpartition(row, -top_k)[-top_k:]
        else:
            keep = np.arange(J)
        vals = row[keep]
        mask = vals > prune
        keep = keep[mask]
        vals = vals[mask]
        s = vals.sum()
        if s <= 1e-9:
            j = int(np.argmax(row))
            weights[names[j]][str(v)] = 1.0
            continue
        vals = vals / s
        for j, wv in zip(keep, vals):
            weights[names[int(j)]][str(v)] = float(wv)
    return weights


# --------------------------------------------------------------------------- #
# Model drivers
# --------------------------------------------------------------------------- #
def _run(cmd, cwd=None, timeout=1800, env=None, expect=None):
    print(f"[ai-weights] $ {' '.join(cmd)} (cwd={cwd})", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], flush=True)
        # Tools that embed Blender's `bpy` (our helpers and UniRig's extractor)
        # can SIGSEGV (-11) during interpreter teardown AFTER fully writing their
        # output. Treat a non-zero exit as success when the expected artifact
        # actually exists and is non-empty.
        if expect and os.path.exists(expect) and os.path.getsize(expect) > 0:
            print(
                f"[ai-weights] WARN: exit {proc.returncode} but {expect} "
                f"exists ({os.path.getsize(expect)} bytes) — continuing",
                flush=True,
            )
            return proc
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-1500:]}")
    return proc


def _run_unirig(in_glb, work, timeout):
    out_fbx = os.path.join(work, "skin.fbx")
    # generate_skin.sh chains extract.sh + run.py via `eval` and then always
    # `echo done` (exit 0) regardless of run.py's result, so a crash inside
    # run.py shows up only as a MISSING output file. Capture the subprocess
    # logs and surface their tail so the real error reaches the job result.
    proc = _run(
        ["bash", "launch/inference/generate_skin.sh",
         "--input", in_glb, "--output", out_fbx,
         "--faces_target_count", str(FACES_TARGET)],
        cwd=UNIRIG_DIR, timeout=timeout, expect=out_fbx,
    )
    if not os.path.exists(out_fbx):
        raise RuntimeError(
            f"UniRig produced no skin.fbx (generate_skin.sh exit {proc.returncode}).\n"
            f"--- stdout tail ---\n{(proc.stdout or '')[-3000:]}\n"
            f"--- stderr tail ---\n{(proc.stderr or '')[-3000:]}"
        )
    return out_fbx, "AI_UNIRIG"


def _run_skintokens(in_glb, work, timeout):
    out_glb = os.path.join(work, "skin.glb")
    proc = _run(
        [PYBIN, "demo.py", "--input", in_glb, "--output", out_glb,
         "--use_skeleton", "--use_transfer"],
        cwd=SKINTOKENS_DIR, timeout=timeout, expect=out_glb,
    )
    if not os.path.exists(out_glb):
        raise RuntimeError(
            f"SkinTokens produced no skin.glb (demo.py exit {proc.returncode}).\n"
            f"--- stdout tail ---\n{(proc.stdout or '')[-3000:]}\n"
            f"--- stderr tail ---\n{(proc.stderr or '')[-3000:]}"
        )
    return out_glb, "AI_SKINTOKENS"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def compute(data, model, timeout):
    t0 = time.time()
    verts = data["vertices"]
    tris = data["triangles"]
    bones = data["bones"]
    N = len(verts)
    J = len(bones)
    bone_names = [b["name"] for b in bones]
    print(f"[ai-weights] model={model} verts={N} tris={len(tris)} bones={J}", flush=True)

    with tempfile.TemporaryDirectory() as work:
        # 1. mesh + skeleton → JSON for the bpy builder
        mesh_json = os.path.join(work, "mesh.json")
        with open(mesh_json, "w") as f:
            json.dump({
                "vertices": verts.tolist(),
                "triangles": tris.tolist(),
                "bones": bones,
            }, f)

        # 2. build input GLB (mesh + armature) via Blender (bpy subprocess)
        in_glb = os.path.join(work, "input.glb")
        t_build = time.time()
        _run([PYBIN, os.path.join(HERE, "blender_build_input.py"), mesh_json, in_glb], timeout=600, expect=in_glb)
        build_s = round(time.time() - t_build, 2)

        # 3. run the chosen skinning model
        t_infer = time.time()
        if model == "skintokens":
            result_path, method = _run_skintokens(in_glb, work, timeout)
        else:
            result_path, method = _run_unirig(in_glb, work, timeout)
        infer_s = round(time.time() - t_infer, 2)

        # 4. read the skinned result (fbx/glb): verts + per-bone vertex groups +
        # the result's own bone head/tail positions (consistent in ONE frame).
        skin_json = os.path.join(work, "skin.json")
        _run([PYBIN, os.path.join(HERE, "blender_read_skin.py"), result_path, skin_json], timeout=600, expect=skin_json)
        with open(skin_json) as f:
            sk = json.load(f)
        src_verts = np.asarray(sk["vertices"], dtype=np.float32)
        name_to_col = {nm: i for i, nm in enumerate(bone_names)}
        src_dense = np.zeros((len(src_verts), J), dtype=np.float32)
        matched = 0
        for gname, col in sk["groups"].items():
            if gname in name_to_col:
                matched += 1
                src_dense[:, name_to_col[gname]] = np.asarray(col, dtype=np.float32)
        src_joints = sk.get("joints", {})

        # quality of the RAW prediction, in its own output frame (before our
        # transfer) — isolates model/skeleton-feeding quality from transfer error
        src_quality = _dominant_quality(src_verts, src_dense, bone_names, src_joints)
        print(f"[ai-weights] src_quality={src_quality} matched={matched}/{J} src_verts={len(src_verts)}", flush=True)

    # our input skeleton (same frame as our mesh) — the anchor for registering
    # the model's proxy back onto our mesh, and the ground truth for dst_quality.
    our_joints = {b["name"]: {"head": b["head"], "tail": b["tail"]}
                  for b in bones if b.get("head") and b.get("tail")}

    # 5. transfer onto our full-res verts (skeleton-anchored registration)
    t_xfer = time.time()
    dst_dense, orient = _transfer_weights(src_verts, src_dense, verts, src_joints, our_joints)
    weights = _densify_topk_normalize(dst_dense, bone_names)
    xfer_s = round(time.time() - t_xfer, 2)
    print(f"[ai-weights] orientation={orient}", flush=True)

    # definitive metric: post-transfer quality on OUR mesh, scored against OUR
    # input skeleton (same frame, no round-trip) — directly comparable to
    # src_quality so we can see how much the transfer degrades the prediction.
    dst_quality = _dominant_quality(verts, dst_dense, bone_names, our_joints)
    print(f"[ai-weights] dst_quality={dst_quality}", flush=True)

    elapsed = round(time.time() - t0, 2)
    bones_with = sum(1 for nm in bone_names if len(weights[nm]) > 0)
    return {
        "weights": weights,
        "bone_count": J,
        "weight_method": method,
        "diagnostics": {
            "input_verts": N,
            "input_tris": int(len(tris)),
            "result_verts": int(len(src_verts)),
            "matched_groups": matched,
            "bones_with_weights": bones_with,
            "model": model,
            "orientation": orient,
            "src_quality": src_quality,
            "dst_quality": dst_quality,
            "timing": {
                "build_glb_s": build_s,
                "inference_s": infer_s,
                "transfer_s": xfer_s,
                "total_s": elapsed,
            },
        },
        "elapsed": elapsed,
    }


def handler(job):
    t0 = time.time()
    ji = job.get("input", {}) or {}
    try:
        data = _load_input(ji)
    except Exception as e:
        return {"error": f"input load failed: {e}", "traceback": traceback.format_exc()}

    for k in ("vertices", "triangles", "bones"):
        if k not in data:
            return {"error": f"missing '{k}' in mesh input"}

    model = "skintokens" if ji.get("model") == "skintokens" else "unirig"
    timeout = int(ji.get("timeout", DEFAULT_TIMEOUT))

    try:
        result_obj = compute(data, model, timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"model timed out after {timeout}s"}
    except Exception as e:
        return {"error": f"ai-weights failed: {e}", "traceback": traceback.format_exc()}

    out_bytes = json.dumps(result_obj).encode("utf-8")

    out_put_url = ji.get("output_put_url")
    if out_put_url:
        ct = ji.get("output_content_type", "application/octet-stream")
        gz = gzip.compress(out_bytes, 6)
        try:
            _put(out_put_url, gz, ct)
        except Exception as e:
            return {"error": f"output upload failed: {e}", "traceback": traceback.format_exc()}
        return {
            "ok": True,
            "output_uploaded": True,
            "output_bytes": len(gz),
            "output_gzip": True,
            "weight_method": result_obj.get("weight_method"),
            "bone_count": result_obj.get("bone_count"),
            "diagnostics": result_obj.get("diagnostics"),
            "elapsed": result_obj.get("elapsed"),
            "handler_elapsed": round(time.time() - t0, 2),
        }

    if len(out_bytes) > MAX_INLINE_OUTPUT:
        return {"error": f"weights too large to return inline ({len(out_bytes)} bytes); provide output_put_url"}
    result_obj["handler_elapsed"] = round(time.time() - t0, 2)
    return result_obj


print("[ai-weights] worker starting...", flush=True)
runpod.serverless.start({"handler": handler})
