# Gemma Kinetic
  
---
 
## Pipeline      
 
 
iPhone (voice) → speech_to_text → Gemma 4 E2B via llama.cpp (GCP T4) → JSON → Flask server → SmolVLA policy (better vla for lerobot arms and cuz im not trying to train manually) → LeRobot arm
  
(Replace with cool visual later) 


---

## Stack

- **Flutter** — iOS app with hold-to-record voice input
- **Gemma 4 E2B** — converts voice transcripts to structured robot instructions, running locally via llama.cpp on a GCP T4 GPU
- **Flask** — receives and saves instructions on laptop
- **SmolVLA** - VLA
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
Imma have this running for the next week or two

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
Then run via Xcode on iPhone.
You need to install Xcode

---

