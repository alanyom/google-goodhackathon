"""
collect_demos.py  —  dual SO-101 snap-cap pill bottle demo collection

Episode flow
────────────
  APPROACH    Both arms move from keyframe wrist pos → hover above targets.
              Long phase + gentle alpha so the arm unfolds gradually.
  ARM2_LOWER  Left arm descends to bottle-body grip height.
  ARM2_GRIP   Left gripper closes — bottle stabilised.
  ARM1_LOWER  Right arm descends to cap top.
  ARM1_GRIP   Right gripper closes on cap.
  PUSH_DOWN   Push cap down 6 mm to defeat snap lock.
  PULL_CAP    Pull cap straight up; cap_joint driven open.
  DONE        Episode ends.
"""

import os
import json
import argparse
import numpy as np
import mujoco
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "use the left arm to stabilise the orange pill bottle by gripping "
    "from the side, then use the right arm to grip the white snap cap, "
    "push down firmly to defeat the snap lock, "
    "and pull the cap straight up to remove it"
)

SCENE_XML = "/home/alanyomedu/SO-ARM100/Simulation/SO101/so101_pill_bottle_dual.xml"

IMG_H, IMG_W = 128, 128
VID_H, VID_W = 480, 640

# ── Bottle randomisation ──────────────────────────────────────────────────
#BOTTLE_X_RANGE = (-0.07,  0.07)
#BOTTLE_Y_RANGE = ( 0.,  0.52)

BOTTLE_X = 0.00
BOTTLE_Y = 0.44

BOTTLE_Z       = 0.062

CAP_JOINT_MAX  = 0.040
SUCCESS_THRESH = 0.028

# ── Arm geometry ──────────────────────────────────────────────────────────
# Left arm grips the bottle from the -X side with a HORIZONTAL approach:
#   hover tip 7 cm to the left at grip height, then slide in to 1 cm left.
ARM2_X_OFFSET       =  0.015   # tip at grip: 1.5 cm right of bottle centre (jaws wrap past centre)
ARM2_HOVER_X_OFFSET = -0.12   # tip at hover: 8.5 cm left (clear of bottle)
ARM2_GRIP_Z         =  0.010   # grip height relative to bottle centre
WRIST_Z_FLOOR       =  0.05    # minimum safe gripper-body Z (keeps arm above table)

# ── Actuator indices ──────────────────────────────────────────────────────
ARM1_GRIP_IDX       = 5
ARM2_GRIP_IDX       = 11
ARM1_WRIST_ROLL_IDX = 4   # ctrl index: right-arm wrist_roll
ARM2_WRIST_ROLL_IDX = 10  # ctrl index: left-arm  l_wrist_roll

# Gripper ctrlrange is [-0.17453, 1.74533]
GRIP_OPEN  =  1.50   # well within open range
GRIP_CLOSE =  0.0    # closed (let spring clamp it)

# Wrist-roll target — rotates gripper jaws 90° to side-grip orientation.
# Tune if jaws are mis-aligned: try +π/2, -π/2, or 0.
GRIP_ROLL_ARM1 =  np.pi / 2   # right arm: cap removal from above
GRIP_ROLL_ARM2 =  np.pi / 2  # left arm:  bottle-body side grip

# ── Simulation timing ─────────────────────────────────────────────────────
SIM_DT         = 0.002
CTRL_HZ        = 20
STEPS_PER_CTRL = int(1 / (CTRL_HZ * SIM_DT))   # 25
MAX_EP_STEPS   = 650

ARM1_EE_BODY = "gripper"
ARM2_EE_BODY = "l_gripper"

CAM_A = "arm1_overhead"
CAM_B = "arm2_overhead"

SETTLE_STEPS = 100   # hold keyframe pose before policy starts


# ── Helpers ───────────────────────────────────────────────────────────────

def lerp(current: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    return current + alpha * (target - current)


def get_mocap_idx(model, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return model.body_mocapid[bid]


def get_site_pos(model, data, site_name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[sid].copy()


def get_body_pos(model, data, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[bid].copy()


def get_cap_joint_addr(model) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cap_joint")
    return model.jnt_qposadr[jid]


def render_cam(renderer, data, cam_name: str, h: int = IMG_H, w: int = IMG_W):
    renderer.update_scene(data, camera=cam_name)
    return cv2.resize(renderer.render(), (w, h))


def configure_welds(model):
    """Zero out the baked-in anchors on both mocap connect constraints.

    MuJoCo computes connect anchors from the zero-joint model configuration.
    Zeroing both anchors makes the constraint enforce:
        gripper_body_origin == mocap_target_origin  (pure position match, no offset).
    """
    for eq_name in ("arm1_mocap_weld", "arm2_mocap_weld"):
        eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
        if eq_id < 0:
            raise RuntimeError(f"Equality '{eq_name}' not found in model")
        # eq_data layout for connect: [anchor_body1(3), anchor_body2(3)]
        model.eq_data[eq_id, 0:6] = 0.0   # both anchors at body origins


def set_bottle_pose(model, data, x: float, y: float):
    jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bottle_joint")
    addr = model.jnt_qposadr[jid]
    data.qpos[addr:addr + 3]     = [x, y, BOTTLE_Z]
    data.qpos[addr + 3:addr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)


def reset_to_home(model, data, bottle_x: float, bottle_y: float,
                  verbose: bool = False):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    set_bottle_pose(model, data, bottle_x, bottle_y)
    mujoco.mj_forward(model, data)

    # Match ctrl to keyframe joints so position controllers hold the pose.
    data.ctrl[:12] = data.qpos[:12]

    data.ctrl[6] += np.radians(5)

    # Seed mocap at current wrist positions.
    # configure_welds() zeroed the relpose, so the weld is immediately satisfied
    # with no constraint violation when physics starts.
    a1_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM1_EE_BODY)
    a2_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM2_EE_BODY)
    A1 = get_mocap_idx(model, "mocap_target")
    A2 = get_mocap_idx(model, "l_mocap_target")
    data.mocap_pos[A1]  = data.xpos[a1_bid].copy()
    data.mocap_pos[A2]  = data.xpos[a2_bid].copy()
    data.mocap_quat[A1] = data.xquat[a1_bid].copy()
    data.mocap_quat[A2] = data.xquat[a2_bid].copy()

    if verbose:
        print(f"  ARM1 wrist world pos : {data.xpos[a1_bid]}")
        print(f"  ARM2 wrist world pos : {data.xpos[a2_bid]}")

    # Brief physics settle — keeps the arm at the keyframe pose.
    for _ in range(50):
        mujoco.mj_step(model, data)


# ── Policy ────────────────────────────────────────────────────────────────

class DualArmSnapCapPolicy:
    """
    Phase-driven cooperative policy.

    Starts from the 'home' keyframe (compact upright folded pose).
    APPROACH uses a very gentle alpha over many steps so the arm unfolds
    gradually without fighting joint limits.

    Phase        Steps   Action
    ──────────── ─────   ──────────────────────────────────────────
    APPROACH      120    Wrist pos → hover above respective targets.
                         alpha=0.04 — slow, smooth unfold.
    ARM2_LOWER     60    Left arm descends to bottle-body grip height.
    ARM2_GRIP      35    Left gripper closes.
    ARM1_LOWER     60    Right arm descends to cap top.
    ARM1_GRIP      35    Right gripper closes on cap.
    PUSH_DOWN      40    Push cap down 6 mm.
    PULL_CAP      150    Pull cap up; drive cap_joint open.
    DONE            —    Terminate.
    """

    APPROACH   = 0
    ARM2_LOWER = 1
    ARM2_GRIP  = 2
    ARM1_LOWER = 3
    ARM1_GRIP  = 4
    PUSH_DOWN  = 5
    PULL_CAP   = 6
    DONE       = 7

    PHASE_STEPS = {
        0: 140,   # APPROACH   — long to allow gradual unfold
        1: 110,   # ARM2_LOWER — longer slide-in: bigger X travel now
        2:  35,   # ARM2_GRIP
        3:  60,   # ARM1_LOWER
        4:  35,   # ARM1_GRIP
        5:  40,   # PUSH_DOWN
        6: 150,   # PULL_CAP
    }

    def __init__(self, model):
        self._cap_addr = get_cap_joint_addr(model)
        self._A1 = get_mocap_idx(model, "mocap_target")
        self._A2 = get_mocap_idx(model, "l_mocap_target")
        # Body / site IDs for dynamic tip-offset compensation
        self._w1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM1_EE_BODY)
        self._w2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ARM2_EE_BODY)
        self._s1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self._s2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "l_gripperframe")
        self.reset()

    def reset(self):
        self.state      = self.APPROACH
        self.phase_step = 0
        self.cap_disp   = 0.0

    def step(self, data, bottle_pos: np.ndarray, cap_pos: np.ndarray):
        A1, A2 = self._A1, self._A2

        # Sync shoulder/elbow/wrist_flex to current qpos so kp=998 doesn't fight the weld.
        # Exclude wrist_roll (indices 4, 10) — those are driven explicitly below.
        data.ctrl[:ARM1_WRIST_ROLL_IDX]                          = data.qpos[:ARM1_WRIST_ROLL_IDX]
        data.ctrl[ARM1_WRIST_ROLL_IDX+1:ARM1_GRIP_IDX]           = data.qpos[ARM1_WRIST_ROLL_IDX+1:ARM1_GRIP_IDX]
        data.ctrl[ARM1_GRIP_IDX+1:ARM2_WRIST_ROLL_IDX]           = data.qpos[ARM1_GRIP_IDX+1:ARM2_WRIST_ROLL_IDX]
        data.ctrl[ARM2_WRIST_ROLL_IDX+1:ARM2_GRIP_IDX]           = data.qpos[ARM2_WRIST_ROLL_IDX+1:ARM2_GRIP_IDX]

        # Gradually rotate both wrists to the side-grip orientation.
        data.ctrl[ARM1_WRIST_ROLL_IDX] = lerp(data.ctrl[ARM1_WRIST_ROLL_IDX], GRIP_ROLL_ARM1, 0.06)
        data.ctrl[ARM2_WRIST_ROLL_IDX] = lerp(data.ctrl[ARM2_WRIST_ROLL_IDX], GRIP_ROLL_ARM2, 0.06)

        # ── Gripper-tip offset compensation ──────────────────────────
        # The mocap drives the WRIST; the fingertips are ~10 cm away.
        # Compute the current world-frame offset from wrist origin to
        # gripperframe site, then subtract it so the TIP reaches the target.
        tip1_off = data.site_xpos[self._s1] - data.xpos[self._w1]
        tip2_off = data.site_xpos[self._s2] - data.xpos[self._w2]

        # ── Per-step derived targets (all in GRIPPER-TIP space) ───────

        # Left arm tip targets — horizontal approach from -X:
        #   hover is to the LEFT at grip height; ARM2_LOWER slides in sideways.
        arm2_grip_tip  = np.array([
            bottle_pos[0] + ARM2_X_OFFSET,
            bottle_pos[1],
            bottle_pos[2] + ARM2_GRIP_Z,
        ])
        arm2_hover_tip = np.array([
            bottle_pos[0] + ARM2_HOVER_X_OFFSET,   # far left, same height
            bottle_pos[1] - 0.09,
            bottle_pos[2] + ARM2_GRIP_Z + 0.015,   # tiny clearance above grip
        ])

        # Right arm tip targets
        arm1_hover_tip = cap_pos + np.array([0, 0, 0.12])
        arm1_touch_tip = cap_pos + np.array([0, 0, 0.002])
        arm1_push_tip  = cap_pos - np.array([0, 0, 0.006])
        arm1_pull_tip  = cap_pos + np.array([0, 0, 0.10])

        # Convert tip targets → wrist (mocap) targets
        arm2_grip  = arm2_grip_tip  - tip2_off
        arm2_hover = arm2_hover_tip - tip2_off
        arm1_hover = arm1_hover_tip - tip1_off
        arm1_touch = arm1_touch_tip - tip1_off
        arm1_push  = arm1_push_tip  - tip1_off
        arm1_pull  = arm1_pull_tip  - tip1_off

        # Clamp wrist Z so the arm never drives itself into the table.
        for tgt in (arm2_grip, arm2_hover, arm1_hover, arm1_touch, arm1_push, arm1_pull):
            tgt[2] = max(tgt[2], WRIST_Z_FLOOR)

        # ── Phase actions ─────────────────────────────────────────────

        if self.state == self.APPROACH:
            # Gentle alpha — arm unfolds smoothly from keyframe pose
            data.mocap_pos[A1] = lerp(data.mocap_pos[A1], arm1_hover,  0.04)
            data.mocap_pos[A2] = lerp(data.mocap_pos[A2], arm2_hover,  0.04)
            data.ctrl[ARM1_GRIP_IDX] = GRIP_OPEN
            data.ctrl[ARM2_GRIP_IDX] = GRIP_OPEN

        elif self.state == self.ARM2_LOWER:
            data.mocap_pos[A2] = lerp(data.mocap_pos[A2], arm2_grip,  0.06)
            data.ctrl[ARM1_GRIP_IDX] = GRIP_OPEN
            data.ctrl[ARM2_GRIP_IDX] = GRIP_OPEN

        elif self.state == self.ARM2_GRIP:
            data.ctrl[ARM1_GRIP_IDX] = GRIP_OPEN
            data.ctrl[ARM2_GRIP_IDX] = GRIP_CLOSE

        elif self.state == self.ARM1_LOWER:
            data.mocap_pos[A1] = lerp(data.mocap_pos[A1], arm1_touch, 0.06)
            data.ctrl[ARM1_GRIP_IDX] = GRIP_OPEN
            data.ctrl[ARM2_GRIP_IDX] = GRIP_CLOSE

        elif self.state == self.ARM1_GRIP:
            data.ctrl[ARM1_GRIP_IDX] = GRIP_CLOSE
            data.ctrl[ARM2_GRIP_IDX] = GRIP_CLOSE

        elif self.state == self.PUSH_DOWN:
            data.mocap_pos[A1] = lerp(data.mocap_pos[A1], arm1_push,  0.05)
            data.ctrl[ARM1_GRIP_IDX] = GRIP_CLOSE
            data.ctrl[ARM2_GRIP_IDX] = GRIP_CLOSE

        elif self.state == self.PULL_CAP:
            data.mocap_pos[A1] = lerp(data.mocap_pos[A1], arm1_pull,  0.035)
            self.cap_disp = min(self.cap_disp + 0.0004, CAP_JOINT_MAX)
            data.qpos[self._cap_addr] = self.cap_disp
            data.ctrl[ARM1_GRIP_IDX] = GRIP_CLOSE
            data.ctrl[ARM2_GRIP_IDX] = GRIP_CLOSE

        # ── Advance phase ─────────────────────────────────────────────
        self.phase_step += 1
        if (self.state < self.DONE
                and self.phase_step >= self.PHASE_STEPS.get(self.state, 0)):
            self.state      += 1
            self.phase_step  = 0

    @property
    def done(self) -> bool:    return self.state == self.DONE
    @property
    def success(self) -> bool: return self.cap_disp >= SUCCESS_THRESH
    @property
    def phase_name(self) -> str:
        return {
            self.APPROACH:   "APPROACH",
            self.ARM2_LOWER: "ARM2_LOWER",
            self.ARM2_GRIP:  "ARM2_GRIP",
            self.ARM1_LOWER: "ARM1_LOWER",
            self.ARM1_GRIP:  "ARM1_GRIP",
            self.PUSH_DOWN:  "PUSH_DOWN",
            self.PULL_CAP:   "PULL_CAP",
            self.DONE:       "DONE",
        }.get(self.state, "?")


# ── Dataset writer ────────────────────────────────────────────────────────

class DatasetWriter:
    def __init__(self, output_dir: str, task: str):
        self.out      = Path(output_dir)
        self.task     = task
        self.frames:   list[dict] = []
        self.episodes: list[dict] = []
        self.ep_idx   = 0
        for cam in (CAM_A, CAM_B):
            (self.out / "videos" / cam).mkdir(parents=True, exist_ok=True)
        (self.out / "data").mkdir(parents=True, exist_ok=True)

    def start_episode(self):
        self.ep_frames: list[dict] = []
        self.ep_cam_a:  list       = []
        self.ep_cam_b:  list       = []

    def add_frame(self, data, img_a, img_b):
        self.ep_frames.append({
            "episode_index":      self.ep_idx,
            "frame_index":        len(self.ep_frames),
            "timestamp":          len(self.ep_frames) / CTRL_HZ,
            "task":               self.task,
            "observation.state":  data.qpos[0:12].tolist(),
            "action":             data.ctrl[0:12].tolist(),
            "next.done":          False,
            "next.success":       False,
        })
        self.ep_cam_a.append(img_a)
        self.ep_cam_b.append(img_b)

    def end_episode(self, success: bool):
        if not self.ep_frames:
            return
        self.ep_frames[-1]["next.done"]    = True
        self.ep_frames[-1]["next.success"] = success
        for cam, imgs in ((CAM_A, self.ep_cam_a), (CAM_B, self.ep_cam_b)):
            d = self.out / "videos" / cam / f"ep{self.ep_idx:05d}"
            d.mkdir(exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(str(d / f"{i:05d}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frames.extend(self.ep_frames)
        self.episodes.append({
            "episode_index": self.ep_idx,
            "tasks":   [self.task],
            "length":  len(self.ep_frames),
            "success": success,
        })
        self.ep_idx += 1

    def finalize(self):
        pq.write_table(
            pa.Table.from_pylist(self.frames),
            str(self.out / "data" / "train-00000-of-00001.parquet"))
        with open(self.out / "episodes.jsonl", "w") as f:
            for ep in self.episodes:
                f.write(json.dumps(ep) + "\n")
        with open(self.out / "meta.json", "w") as f:
            json.dump({
                "codebase_version": "v3.0",
                "robot_type":       "so101_dual",
                "fps":              CTRL_HZ,
                "tasks":            [self.task],
                "total_episodes":   len(self.episodes),
                "total_frames":     len(self.frames),
                "obs_dim":          12,
                "action_dim":       12,
                "cameras":          [CAM_A, CAM_B],
                "start_pose":       "home_keyframe",
            }, f, indent=2)
        n_ok = sum(e["success"] for e in self.episodes)
        print(f"\n✓  Dataset → {self.out}")
        print(f"   Episodes : {len(self.episodes)}")
        print(f"   Frames   : {len(self.frames)}")
        print(f"   Success  : {n_ok}/{len(self.episodes)}"
              f"  ({100*n_ok/max(len(self.episodes),1):.1f}%)")


# ── Preview ───────────────────────────────────────────────────────────────

def record_preview(model, data, output_path: Path, bottle_xy=(BOTTLE_X, BOTTLE_Y)):
    print("Recording preview …")
    reset_to_home(model, data, *bottle_xy, verbose=True)
    # One renderer shared between both cameras (renders one at a time).
    renderer = mujoco.Renderer(model, height=VID_H, width=VID_W)
    writer   = cv2.VideoWriter(str(output_path),
                               cv2.VideoWriter_fourcc(*"mp4v"),
                               CTRL_HZ, (VID_W * 2, VID_H))
    policy = DualArmSnapCapPolicy(model)

    for step_i in range(MAX_EP_STEPS):
        bottle_pos = get_body_pos(model, data, "bottle")
        cap_pos    = get_site_pos(model, data, "cap_center")
        policy.step(data, bottle_pos, cap_pos)
        for _ in range(STEPS_PER_CTRL):
            mujoco.mj_step(model, data)

        img_side = render_cam(renderer, data, "preview_cam",   VID_H, VID_W)
        img_top  = render_cam(renderer, data, "overhead_cam",  VID_H, VID_W)
        frame = np.concatenate([img_side, img_top], axis=1)   # 480 × 1280, RGB

        label = f"Phase: {policy.phase_name}  step {step_i:03d}"
        cv2.putText(frame, label,
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 80), 2)
        cv2.putText(frame, "SIDE VIEW",
                    (10, VID_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, "TOP VIEW",
                    (VID_W + 10, VID_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if policy.done:
            break

    renderer.close()
    writer.release()
    result = "SUCCESS ✓" if policy.success else "FAILED ✗"
    print(f"✓  Preview → {output_path}  [{result}]")


# ── Collect ───────────────────────────────────────────────────────────────

def collect(num_demos: int, output_dir: str, preview: bool):
    print("Loading model …")
    model    = mujoco.MjModel.from_xml_path(SCENE_XML)
    configure_welds(model)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    policy   = DualArmSnapCapPolicy(model)
    writer   = DatasetWriter(output_dir, TASK_DESCRIPTION)
  #  rng      = np.random.default_rng(42)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if preview:
        record_preview(model, data, Path(output_dir) / "preview.mp4")

    for ep in tqdm(range(num_demos), desc="Collecting demos"):
        #bx = float(rng.uniform(*BOTTLE_X_RANGE))
        #by = float(rng.uniform(*BOTTLE_Y_RANGE))

        bx = BOTTLE_X
        by = BOTTLE_Y

        reset_to_home(model, data, bx, by)
        policy.reset()
        writer.start_episode()

        for _ in range(MAX_EP_STEPS):
            bottle_pos = get_body_pos(model, data, "bottle")
            cap_pos    = get_site_pos(model, data, "cap_center")
            policy.step(data, bottle_pos, cap_pos)
            for _ in range(STEPS_PER_CTRL):
                mujoco.mj_step(model, data)
            img_a = render_cam(renderer, data, CAM_A)
            img_b = render_cam(renderer, data, CAM_B)
            writer.add_frame(data, img_a, img_b)
            if policy.done:
                break

        writer.end_episode(policy.success)

    writer.finalize()
    renderer.close()


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    p = argparse.ArgumentParser()
    p.add_argument("--num-demos",  type=int, default=200)
    p.add_argument("--output-dir", type=str, default="~/data/pill_demos_dual")
    p.add_argument("--preview",    action="store_true")
    args = p.parse_args()
    collect(args.num_demos, os.path.expanduser(args.output_dir), args.preview)