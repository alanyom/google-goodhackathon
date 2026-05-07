import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("/home/alanyomedu/SO-ARM100/Simulation/SO101/so101_pill_bottle_dual.xml")
data  = mujoco.MjData(model)

# Check joint order
print("=== Joint names and qpos addresses ===")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    addr = model.jnt_qposadr[i]
    print(f"  joint[{i}] addr={addr:2d}  {name}")

# Load keyframe
kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, kid)
mujoco.mj_forward(model, data)

print("\n=== qpos after keyframe ===")
print(data.qpos[:20])

print("\n=== ctrl after keyframe (before fix) ===")
print(data.ctrl[:12])

data.ctrl[:12] = data.qpos[:12]
print("\n=== ctrl after fix ===")
print(data.ctrl[:12])

print("\n=== actuator names ===")
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  ctrl[{i}]  {name}")

wrist_bid   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
l_wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wrist")

print("\n=== Wrist positions after keyframe ===")
print("Right wrist pos:", data.xpos[wrist_bid])
print("Left  wrist pos:", data.xpos[l_wrist_bid])
print("Bottle is at:    [0.0, 0.42, 0.062]")
print("Cap is at:       [0.0, 0.42, 0.117]")