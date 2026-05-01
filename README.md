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

##The Problem

For elderly and disabled individuals, performing basic physical tasks like opening a pill bottle, picking up an object, or pressing a button can be a significant barrier to independence. Existing robotic assistants either require complex interfaces or depend on cloud AI that raises legitimate privacy concerns. A device that is always listening and sending data to a remote server is not something many people want in their bedroom or bathroom.
Gemma Kinetic is a voice-controlled robotic arm assistant that is private by design. All AI inference runs on hardware the user controls. No audio, no video, and no commands leave the system unless explicitly configured to do so.

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
