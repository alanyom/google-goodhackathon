import json
import glob
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# load whatever policy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.eval()


files = sorted(glob.glob("output/*.json"))
latest = files[-1]

with open(latest) as f:
    data = json.load(f)

task_description = data["task_description"]
print(f"Task: {task_description}")

# replace with real camera frame when arm arrives
# for now just confirm the pipeline loads and reads correctly
print("Bridge script ready — waiting for arm.")