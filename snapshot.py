"""
diag.py — render one frame from every camera and save as PNGs
Usage: MUJOCO_GL=egl python diag.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import cv2
import numpy as np

SCENE_XML = "/home/alanyomedu/SO-ARM100/Simulation/SO101/so101_pill_bottle_dual.xml"
OUT_DIR   = os.path.expanduser("~/data/diag")
os.makedirs(OUT_DIR, exist_ok=True)

model    = mujoco.MjModel.from_xml_path(SCENE_XML)
data     = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)
mujoco.mj_forward(model, data)

mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

wrist_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
l_wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wrist")

print("wrist pos:  ", data.xpos[wrist_id])
print("l_wrist pos:", data.xpos[l_wrist_id])

# Put bottle at centre of workspace so it's visible
jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")
addr = model.jnt_qposadr[jid]
data.qpos[addr:addr+3] = [0.0, 0.42, 0.062]
data.qpos[addr+3:addr+7] = [1, 0, 0, 0]
mujoco.mj_forward(model, data)

cameras = ["arm1_overhead", "arm2_overhead", "overhead_cam", "preview_cam"]

for cam in cameras:
    try:
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        path = os.path.join(OUT_DIR, f"{cam}.png")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        # Check if image is all black
        brightness = np.mean(img)
        print(f"{cam}: brightness={brightness:.1f}  → {path}")
    except Exception as e:
        print(f"{cam}: ERROR — {e}")

renderer.close()
print(f"\nImages saved to {OUT_DIR}")
print("Copy them here so we can see what each camera sees.")