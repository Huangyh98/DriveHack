"""Bake the Mixamo run-in-place animation into a per-frame mesh sequence.

For each animation frame this writes the deformed character as world-space
vertices (in Blender's Y-up meters), plus the shared faces/UVs and texture
image. The displacement + heading of the character is handled later in Python
(via a rigid transform), so we bake the character standing at the origin,
facing +Y, for every loop frame.

Run with the newer Blender:

    ~/blender/blender-4.4.3-linux-x64/blender --background \
        --python tools/bake_runner_frames.py -- \
        --blend man/AdvSerial_v2_runing_rd.blend \
        --out outputs/assets/runner_seq.npz \
        --frames 40
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser("Bake runner animation to npz")
    parser.add_argument("--blend", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=40, help="number of loop frames to bake")
    args = parser.parse_args(argv)

    import bpy
    import numpy as np

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args.blend))
    sc = bpy.context.scene
    arm = bpy.data.objects["Armature"]
    mesh_objs = [bpy.data.objects[n] for n in ("man", "clothes_1", "pants_1")]
    action = arm.animation_data.action
    a0, a1 = int(action.frame_range[0]), int(action.frame_range[1])
    loop_len = a1 - a0 + 1
    n_frames = args.frames
    print(f"[bake] action {action.name} frames {a0}-{a1} (loop {loop_len}); baking {n_frames} frames")

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Collect static geometry per mesh: faces, uv, material texture.
    meshes_info = []
    for obj in mesh_objs:
        me = obj.data
        uv_layer = me.uv_layers.active.data if me.uv_layers else None
        # Triangulate face indices (nvdiffrast wants triangles).
        tris = []
        uv_per_vertex_loop = []
        for poly in me.polygons:
            idx = poly.loop_indices
            if len(idx) == 3:
                tris.append([me.loops[i].vertex_index for i in idx])
                if uv_layer is not None:
                    uv_per_vertex_loop.append([uv_layer[i].uv for i in idx])
            else:  # fan-triangulate quads/ngons
                start = idx[0]
                for k in range(1, len(idx) - 1):
                    tris.append([
                        me.loops[start].vertex_index,
                        me.loops[idx[k]].vertex_index,
                        me.loops[idx[k + 1]].vertex_index,
                    ])
                    if uv_layer is not None:
                        uv_per_vertex_loop.append([
                            uv_layer[start].uv,
                            uv_layer[idx[k]].uv,
                            uv_layer[idx[k + 1]].uv,
                        ])
        tris = np.asarray(tris, dtype=np.int64)

        # UVs per triangle vertex: shape (F, 3, 2)
        if uv_per_vertex_loop:
            uvs = np.asarray(uv_per_vertex_loop, dtype=np.float32)
        else:
            uvs = np.zeros((len(tris), 3, 2), dtype=np.float32)

        # Extract the base color texture image as uint8 HxWx4.
        tex = None
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            if mat is None or mat.node_tree is None:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    img = node.image
                    pix = np.asarray(img.pixels[:])  # flat RGBA float
                    h, w = img.size[1], img.size[0]
                    pix = pix.reshape(h, w, 4)
                    tex = (pix * 255).clip(0, 255).astype(np.uint8)
                    break
            if tex is not None:
                break
        if tex is None:
            tex = np.full((4, 4, 4), 200, dtype=np.uint8)

        meshes_info.append({"name": obj.name, "faces": tris, "uvs": uvs, "tex": tex, "n_verts": len(me.vertices)})

    # Bake per-frame world-space vertices for each mesh.
    all_verts = {mi["name"]: np.zeros((n_frames, mi["n_verts"], 3), dtype=np.float32) for mi in meshes_info}
    for fi in range(n_frames):
        # Loop the action.
        frame = a0 + (fi % loop_len)
        sc.frame_set(frame)
        depsgraph.update()
        for obj in mesh_objs:
            obj_eval = obj.evaluated_get(depsgraph)
            mw = obj.matrix_world
            me = obj_eval.to_mesh()
            vs = np.empty((len(me.vertices), 3), dtype=np.float32)
            me.vertices.foreach_get("co", vs.ravel())
            # apply world matrix (object is parented to armature at origin, mostly identity,
            # but be correct anyway). mathutils Matrix -> flat 16 floats.
            m = obj.matrix_world
            M = np.array([m[0][0],m[0][1],m[0][2],m[0][3],
                          m[1][0],m[1][1],m[1][2],m[1][3],
                          m[2][0],m[2][1],m[2][2],m[2][3],
                          m[3][0],m[3][1],m[3][2],m[3][3]], dtype=np.float64).reshape(4,4)
            hom = np.concatenate([vs.astype(np.float64), np.ones((len(vs),1),dtype=np.float64)], axis=1)
            vs = (hom @ M.T)[:, :3].astype(np.float32)
            all_verts[obj.name][fi] = vs
            obj_eval.to_mesh_clear()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save = {
        "n_frames": n_frames,
    }
    for mi in meshes_info:
        save[f"{mi['name']}/faces"] = mi["faces"]
        save[f"{mi['name']}/uvs"] = mi["uvs"]
        save[f"{mi['name']}/tex"] = mi["tex"]
        save[f"{mi['name']}/verts"] = all_verts[mi["name"]]
    np.savez_compressed(os.path.abspath(args.out), **save)
    print(f"[bake] wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")
    # report bbox so we can size placement
    v = all_verts["man"][0]
    print(f"[bake] man frame0 bbox y(min/max)={v[:,1].min():.3f}/{v[:,1].max():.3f} "
          f"z(min/max)={v[:,2].min():.3f}/{v[:,2].max():.3f} x(min/max)={v[:,0].min():.3f}/{v[:,0].max():.3f}")


if __name__ == "__main__":
    main()
