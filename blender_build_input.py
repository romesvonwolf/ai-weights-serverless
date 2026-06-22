"""
Build a GLB (mesh + armature) from our mesh+skeleton JSON, for UniRig/SkinTokens.

Run via the bpy python module:
    python blender_build_input.py mesh.json out.glb

mesh.json: { "vertices": [[x,y,z]...], "triangles": [[i,j,k]...],
             "bones": [{ "name", "head":[x,y,z], "tail":[x,y,z], "parent": name|null }] }

The armature carries OUR bone names so the predicted skin weights round-trip
back to the same names. The mesh is bound to the armature (empty groups) so the
glTF exporter writes a skinned skeleton the extractor can read.
"""
import sys
import json
import bpy
from mathutils import Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def build(mesh_json, out_glb):
    with open(mesh_json) as f:
        data = json.load(f)
    verts = [tuple(v) for v in data["vertices"]]
    faces = [tuple(t) for t in data["triangles"]]
    bones = data["bones"]

    reset_scene()

    # --- mesh ---
    mesh = bpy.data.meshes.new("Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh_obj = bpy.data.objects.new("Mesh", mesh)
    bpy.context.collection.objects.link(mesh_obj)

    # --- armature ---
    arm = bpy.data.armatures.new("Armature")
    arm_obj = bpy.data.objects.new("Armature", arm)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    created = {}
    for b in bones:
        eb = arm.edit_bones.new(b["name"])
        eb.head = Vector(b["head"])
        tail = Vector(b["tail"])
        if (tail - eb.head).length < 1e-5:
            tail = eb.head + Vector((0.0, 0.05, 0.0))
        eb.tail = tail
        created[b["name"]] = eb
    for b in bones:
        p = b.get("parent")
        if p and p in created:
            created[b["name"]].parent = created[p]
    bpy.ops.object.mode_set(mode="OBJECT")

    # --- bind mesh to armature (creates empty vertex groups per bone) ---
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    # ARMATURE_NAME = empty groups (no auto weights); we only need the skeleton.
    bpy.ops.object.parent_set(type="ARMATURE_NAME")

    # ensure a vertex group exists for every bone so the skin is exportable
    for b in bones:
        if b["name"] not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=b["name"])

    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.export_scene.gltf(
        filepath=out_glb,
        export_format="GLB",
        use_selection=False,
        export_yup=True,
        export_skins=True,
        export_apply=False,
    )
    print(f"[build_input] wrote {out_glb}: {len(verts)} verts, {len(faces)} faces, {len(bones)} bones", flush=True)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
