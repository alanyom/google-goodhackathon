# Gemma Kinetic

Voice-Controlled Assistive Robotics via llama.cpp
Giving elderly and disabled users a natural language interface to a robotic arm, powered entirely by Gemma 4 running on a T4 GPU via llama.cpp.
---

## Pipeline
![Pipeline](googleflow.svg) 

---

## Stack

- **Flutter** — iOS app with hold-to-record voice input
- **Gemma 4 E2B** — converts voice transcripts to structured robot instructions via llama.cpp on a GCP T4 GPU
- **Flask** — receives and saves instructions locally
- **SmolVLA** — vision-language-action policy model for LeRobot arms
- **LeRobot** — robot arm framework

---

## The Problem

For elderly and disabled individuals, performing basic physical tasks like opening a pill bottle, picking up an object, or pressing a button can be a significant barrier to independence. Existing robotic assistants either require complex interfaces or depend on cloud AI that raises legitimate privacy concerns. A device that is always listening and sending data to a remote server is not something many people want in their bedroom or bathroom.
Gemma Kinetic is a voice-controlled robotic arm assistant that is private by design. All AI inference runs on hardware the user controls. No audio, no video, and no commands leave the system unless explicitly configured to do so.

---

## Impact

The natural language interface lowers the barrier to entry significantly. A user does not need to learn a custom interface or remember specific commands. They describe what they want in plain language and the system handles the mechanics.
The secondary impact is the dataset the system generates. Every voice command paired with its structured Gemma output and the resulting arm execution creates a record of physical intent mapping: how diverse users describe physical tasks, how a language model interprets those descriptions, and what motor actions result. This data has direct value for improving both the language-to-action translation layer and the vision feedback loop over time. It is the kind of dataset that does not exist at scale and that robotics researchers actively need.

---

## Challenges

Getting a language model to reliably drive a physical system introduces problems that do not exist in purely software contexts. A hallucinated JSON field does not crash a web app. On a robot arm, a malformed command can mean a missed grasp, a dropped object, or worse.
The core challenge was building a pipeline where every step is predictable. Gemma 4 needed to produce valid, parseable JSON every time, not just most of the time. SmolVLA needed to receive that JSON in a format it could act on directly, without a translation layer. And the vision feedback loop needed to close fast enough to be useful without saturating the inference budget.
Each of these required deliberate design choices rather than off-the-shelf integration, which is described in the technical sections below.

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
