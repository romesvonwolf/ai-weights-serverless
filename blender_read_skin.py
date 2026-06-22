"""
Read a skinned result (FBX or GLB) → per-vertex vertex-group weights JSON.

Run via the bpy python module:
    python blender_read_skin.py result.fbx out.json
    python blender_read_skin.py result.glb out.json

Output JSON:
  {
    "names":    [bone/group names present on the armature],
    "vertices": [[x,y,z], ...]   # world-space verts of the skinned mesh
    "groups":   { groupName: [w_0, w_1, ..., w_{Ns-1}], ... }  # dense per vertex
    "joints":   { boneName: {"head":[x,y,z], "tail":[x,y,z]} }  # result armature
  }

The caller NN-transfers these (in normalized space) onto our full-res mesh, so
the absolute coordinate frame here does not matter — only relative geometry.
`joints` lets the caller measure skinning quality in the result's OWN space
(dominant-bone centroid vs bone), isolating prediction quality from transfer.
"""
import os
import sys
import json
import bpy


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_any(path):
    if path.lower().endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif path.lower().endswith((".fbx",)):
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported result format: {path}")


def pick_skinned_mesh():
    """Return the mesh object with the most vertex groups (the skinned one)."""
    best = None
    best_groups = -1
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and len(obj.vertex_groups) > best_groups:
            best = obj
            best_groups = len(obj.vertex_groups)
    if best is None:
        raise RuntimeError("no mesh found in result")
    return best


def read_joints():
    """World-space head/tail of every bone in the result armature (for caller-
    side quality diagnostics in the result's own coordinate frame)."""
    out = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "ARMATURE":
            continue
        mw = obj.matrix_world
        for b in obj.data.bones:
            h = mw @ b.head_local
            t = mw @ b.tail_local
            out[b.name] = {"head": [h.x, h.y, h.z], "tail": [t.x, t.y, t.z]}
    return out


def read(result_path, out_json):
    reset_scene()
    load_any(result_path)
    mesh_obj = pick_skinned_mesh()

    mat = mesh_obj.matrix_world
    mesh = mesh_obj.data
    Ns = len(mesh.vertices)

    group_names = [vg.name for vg in mesh_obj.vertex_groups]
    gidx_to_name = {vg.index: vg.name for vg in mesh_obj.vertex_groups}
    groups = {nm: [0.0] * Ns for nm in group_names}

    vertices = []
    for vi, v in enumerate(mesh.vertices):
        co = mat @ v.co
        vertices.append([co.x, co.y, co.z])
        for g in v.groups:
            nm = gidx_to_name.get(g.group)
            if nm is not None:
                groups[nm][vi] = float(g.weight)

    out = {
        "names": group_names,
        "vertices": vertices,
        "groups": groups,
        "joints": read_joints(),
    }
    with open(out_json, "w") as f:
        json.dump(out, f)
    nonzero = sum(1 for nm in group_names if any(w > 0 for w in groups[nm]))
    print(f"[read_skin] {Ns} verts, {len(group_names)} groups ({nonzero} non-empty) → {out_json}", flush=True)


if __name__ == "__main__":
    read(sys.argv[1], sys.argv[2])
    # bpy 4.x reliably SIGSEGVs (exit -11) during Python interpreter teardown on
    # headless workers. The JSON is already fully written above, so flush our
    # logs and hard-exit 0 to stop that bogus crash being treated as a failure.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
