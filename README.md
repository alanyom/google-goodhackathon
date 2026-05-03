# Gemma Kinetic

### Voice-Controlled Assistive Robotics powered by Gemma 4 via llama.cpp
 


## Pipeline
![Pipeline](googleflow.svg) 

---

## The Problem

For elderly and disabled individuals, performing basic physical tasks like opening a pill bottle, picking up an object, or pressing a button can be a significant barrier to independence. Existing robotic assistants either require complex interfaces or depend on cloud AI that raises legitimate privacy concerns. A device that is always listening and sending data to a remote server is not something many people want in their bedroom or bathroom.
Gemma Kinetic is a voice-controlled robotic arm assistant that is private by design. All AI inference runs on hardware the user controls. No audio, no video, and no commands leave the system unless explicitly configured to do so.

---

## Impact

The natural language interface lowers the barrier to entry significantly. A user does not need to learn a custom interface or remember specific commands. They describe what they want in plain language and the system handles the mechanics.
The secondary impact is the dataset the system generates. Every voice command paired with its structured Gemma output and the resulting arm execution creates a record of physical intent mapping: how diverse users describe physical tasks, how a language model interprets those descriptions, and what motor actions result. This data has direct value for improving both the language-to-action translation layer and the vision feedback loop over time. It is the kind of dataset that does not exist at scale and that robotics researchers actively need.

---

## How Gemma 4 Is Used

Gemma 4 E2B handles two distinct roles in the pipeline.
Structured output generation. The primary role is converting a free-form voice transcript into a JSON command with three fields: task_description, object, and action. The system prompt instructs Gemma to respond with valid JSON only, no markdown, no explanation, making the output directly parseable and passable to the robot control layer. A low temperature setting of 0.2 keeps outputs consistent and deterministic, which matters when commanding hardware.

A typical exchange:
- Voice input: "Can you open the water bottle for me, the lid is pretty tight"
- Gemma output:

```
json{
  "task_description": "locate the water bottle, apply firm counterclockwise rotational force to unscrew the cap",
  "object": "water bottle cap",
  "action": "unscrew"
}
```

This shows how Gemma interprets nuance. The user said "pretty tight" and Gemma translated that into "apply firm rotational force", encoding the physical constraint into the instruction without any explicit mapping.
Multimodal scene understanding. The second role uses Gemma 4's native vision capability for the feedback loop described below.

---

## Why These Technical Choices
Gemma 4 E2B over larger models. The E2B model hits a practical sweet spot: strong enough instruction following for reliable structured output, efficient enough to run quantized on a T4 without saturating the GPU, and natively multimodal so the same model handles both the text-to-JSON and vision feedback roles. A larger model would improve output quality marginally but would make deployment on accessible hardware significantly harder.

llama.cpp over hosted inference. The privacy-first constraint is non-negotiable for the target use case. Elderly and disabled users in residential settings should not have their voice commands and camera feeds routed through a third-party API. llama.cpp makes on-premise inference practical without dedicated ML infrastructure.
SmolVLA over manual policy training. Training a robot policy from scratch requires significant hardware, data collection time, and domain expertise. SmolVLA provides a capable pretrained base that accepts natural language task conditioning and integrates directly with Gemma's output format. Fine-tuning SmolVLA on task-specific demonstrations is the natural production path from here.
Flutter for the mobile client. A single codebase targeting iOS natively, using platform speech recognition, and communicating via standard HTTP kept the mobile layer thin and the inference layer independent. The hold-to-record interaction pattern is also intentional: it avoids any always-listening behavior, consistent with the privacy design of the overall system.

---

## Setup

### GCP VM (llama.cpp server)

```bash
/home/llama.cpp/build/bin/llama-server \
  -m /home/models/google_gemma-4-E2B-it-Q4_K_M.gguf \
  --port 8080 \
  --host 0.0.0.0
```

### Laptop (Flask server)

```bash
cd gemmahack
python3 server.py
```

### Flutter app

Add a `.env` file to `gemmaflutter/` with:

```
VM_EXTERNAL_IP=your_gcp_ip
LOCAL_IP=your_laptop_ip
```

Then run via Xcode on iPhone. Requires Xcode installed.
