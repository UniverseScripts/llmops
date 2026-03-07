# Enterprise Edge Inference Node (Local LLMOps)

A hardened, Dockerized edge-inference pipeline capable of routing trans-continental B2B API traffic to highly constrained local hardware (≤ 4GB VRAM) with zero cloud compute costs. 

This repository bypasses the financial overhead of hyperscaler GPU instances (AWS/GCP) by orchestrating a localized quantized LLM matrix alongside a Zero Trust ingress tunnel.

## Architectural Features

1. **Strict State Isolation:** Implements 8-bit quantization (`bitsandbytes`) for `google/flan-t5-base`. Precision LoRA weights are mapped natively to the FastAPI `app.state` on boot, eradicating ASGI event loop paralysis under concurrent requests.
2. **Zero Trust Ingress:** Orchestrates a dual-container matrix utilizing a `cloudflared` daemon to project an outbound-only, HTTPS-secured Quick Tunnel. This masks the physical IP, bypasses residential NAT, and provides global routing out of the box.
3. **Edge Rate Limiting:** Enforces strict hardware boundaries via an internal, in-memory token-bucket middleware that parses `CF-Connecting-IP` headers, preventing node overload.

## Prerequisites

To execute this orchestration matrix, the host machine must possess:
* Docker Engine & Docker Compose (v3.8+)
* NVIDIA Container Toolkit (for GPU passthrough)
* A CUDA-compatible GPU

## Deployment Matrix

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd llmops
   ```
2. **Boot the Orchestration (Detached):**
  ```bash
  docker-compose up -d --build
  ```
3. **Extract the Global Endpoint:**
The Cloudflare Quick Tunnel will dynamically generate a secure HTTPS endpoint on boot. Extract it from the daemon logs:
 ```bash
  docker logs edge-ingress-tunnel
 ```
Look for the URL ending in *.trycloudflare.com.

4. **Verify the Node State:**
Ensure the inference node has finished mapping the tensors into VRAM and opened the ASGI socket:
```bash
  docker logs -f enterprise-inference-node
```
Wait for: INFO: Application startup complete.
