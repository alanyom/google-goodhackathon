# Gemma Kinetic

Voice-controlled assistive robotics powered by Gemma 4 running fully offline via llama.cpp.

---

## Pipeline

---

## Stack

- **Flutter** — iOS app with hold-to-record voice input
- **Gemma 4 E2B** — converts voice transcripts to structured robot instructions via llama.cpp on a GCP T4 GPU
- **Flask** — receives and saves instructions locally
- **SmolVLA** — vision-language-action policy model for LeRobot arms
- **LeRobot** — robot arm framework

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
