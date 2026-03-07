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

## API Usage Protocol
The endpoint strictly requires the trailing slash and an Enterprise Token header.

cURL Extraction Test:
```bash
   curl -X POST "https://<your-generated-url>[.trycloudflare.com/generate/](https://.trycloudflare.com/generate/)" \
        -H "Content-Type: application/json" \
        -H "X-Enterprise-Token: sk_live_edge_node_001" \
        -d '{"prompt": "Explain the dialectics of distributed systems."}'
```
     
## Limitations & Production Scaling
This specific architecture is engineered as a lightweight, zero-cost edge demonstration. If you are migrating this to a production multi-node cluster, you must address the following structural gaps:

* Authentication (`service/auth.py`): Currently utilizes a static Python set for API key validation. Production requires integration with a persistent database (e.g., PostgreSQL/Redis) and dynamic token hashing.

* Rate Limiting (`security/rate_limiter.py`): Operates on a transient, localized ASGI in-memory dictionary. Upon container restart, the limits reset. In a multi-worker production environment, this will cause state fracturing.

## Open-Source Contributions & PRs
This repository is strictly open-source. For infrastructure engineers looking to harden the pipeline, Pull Requests are currently open for the following architectural upgrades:

1. Implementation of a lightweight Redis caching layer to replace the in-memory token bucket.

2. Integration of a stateless authentication router via JWTs or a database engine.
