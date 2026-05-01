import json
import glob
import time
import sys
import random
import numpy as np


POLICY_LOADED = False
try:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.eval()
    POLICY_LOADED = True
    print("[bridge] SmolVLA policy loaded ✓")
except Exception as e:
    print(f"[bridge] SmolVLA not available ({e}) — running in simulation mode")

ARM_CONNECTED = False
try:
    from lerobot.common.robot_devices.robots.factory import make_robot
    from lerobot.common.robot_devices.utils import busy_wait
    robot = make_robot("koch")   
    robot.connect()
    ARM_CONNECTED = True
    print("[bridge] Robot arm connected ✓")
except Exception as e:
    print(f"[bridge] Arm not available ({e}) — motor dispatch will be simulated")



def get_latest_instruction() -> dict | None:
    files = sorted(glob.glob("output/*.json"))
    if not files:
        print("[bridge] No instruction files found in output/")
        return None
    with open(files[-1]) as f:
        return json.load(f)



def get_camera_frame():
    import cv2
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[bridge] Camera read failed")
        return None
    return frame   


def frame_to_tensor(frame) -> torch.Tensor:
    """Convert a raw cv2 frame to the normalized CHW float tensor SmolVLA expects."""
    import cv2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (224, 224))
    tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)   



def run_policy(camera_frame, task_description: str) -> torch.Tensor:
    """
    Runs SmolVLA inference and returns a raw action tensor.
    Shape is typically (1, action_dim) where action_dim = num joints.
    """
    image_tensor = frame_to_tensor(camera_frame)

    if ARM_CONNECTED:
        state = torch.tensor(robot.get_state(), dtype=torch.float32).unsqueeze(0)
    else:
        state = torch.zeros(1, 6)  

    observation = {
        "observation.images.top": image_tensor,
        "observation.state": state,
        "task": [task_description],
    }

    with torch.no_grad():
        action = policy.select_action(observation)  

    return action



def dispatch_real(action_tensor: torch.Tensor):
    """Send a SmolVLA action tensor directly to the LeRobot arm."""
    action = action_tensor.squeeze(0).numpy()
    print(f"[bridge] Sending action to arm: {np.round(action, 3)}")
    robot.send_action(action)
    busy_wait(1 / 30)   # 30 Hz control loop tick



def simulate_action_sequence(instruction: dict) -> list[dict]:
    action = instruction.get("action", "move").lower()
    task   = instruction.get("task_description", "")
    random.seed(hash(task) % 9999)

    joints = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]

    if any(k in action for k in ["pick", "grab", "grasp"]):
        deltas = [30, -45, 20, 10, 0, 60]
    elif any(k in action for k in ["place", "put", "set"]):
        deltas = [-15, 20, -10, -5, 0, -60]
    elif any(k in action for k in ["open", "unscrew", "twist"]):
        deltas = [5, -10, 5, 0, 180, 30]
    elif any(k in action for k in ["push", "press"]):
        deltas = [10, -30, 40, 0, 0, 0]
    else:
        deltas = [random.randint(-20, 20) for _ in joints]

    return [
        {
            "joint": joint,
            "target_delta_deg": round(delta + random.uniform(-3, 3), 2),
            "velocity": round(random.uniform(0.3, 0.8), 2),
            "duration_ms": random.randint(300, 900),
        }
        for joint, delta in zip(joints, deltas)
    ]


def dispatch_simulated(steps: list[dict]):
    print("\n[bridge] Dispatching action sequence (SIMULATED):")
    print(f"  {'Joint':<16} {'Delta (°)':<12} {'Velocity':<12} {'Duration'}")
    print("  " + "-" * 55)
    for step in steps:
        print(f"  {step['joint']:<16} {step['target_delta_deg']:<12} {step['velocity']:<12} {step['duration_ms']}ms")
        time.sleep(step["duration_ms"] / 1000 * 0.1)
    print("\n[bridge] Sequence complete")



def run():
    print("=" * 60)
    print("Gemma Kinetic — Hardware Bridge")
    print(f"  Policy : {'loaded' if POLICY_LOADED else 'not available'}")
    print(f"  Arm    : {'connected' if ARM_CONNECTED else 'not connected'}")
    print(f"  Mode   : {'REAL' if (POLICY_LOADED and ARM_CONNECTED) else 'SIMULATED'}")
    print("=" * 60)

    record = get_latest_instruction()
    if record is None:
        sys.exit(1)

    instruction = record.get("instruction", record)
    print(f"\n[bridge] Latest instruction")
    print(f"  Task   : {instruction.get('task_description', '?')}")
    print(f"  Object : {instruction.get('object', '?')}")
    print(f"  Action : {instruction.get('action', '?')}")

    if POLICY_LOADED and ARM_CONNECTED:
        camera_frame = get_camera_frame()
        if camera_frame is None:
            print("[bridge] No camera frame — falling back to simulation")
        else:
            action_tensor = run_policy(camera_frame, instruction.get("task_description", ""))
            dispatch_real(action_tensor)
            return

    steps = simulate_action_sequence(instruction)
    dispatch_simulated(steps)


if __name__ == "__main__":
    run()
