# Local LLMOps Edge Node

A self-hosted inference node for a quantized model. It runs as a containerised stack on one machine and is reachable over a Cloudflare tunnel rather than a cloud GPU, so there is no hyperscaler bill and no inbound port forwarding. Rate-limiting and API-key state live in Redis and PostgreSQL rather than in process memory, so both survive a worker restart.

Built as a learning exercise in self-hosted LLM serving. It runs on a single machine behind an ephemeral `*.trycloudflare.com` tunnel — not a stable endpoint, and not something anyone is depending on.

## Architectural Matrix

The infrastructure is defined by strict boundary segregation, from GPU tensor allocation up to network ingress.

### Phase 1: The Inference Engine (Compute Layer)
The core compute layer abandons standard API wrapper logic in favor of a locally hosted, quantized model.
* **The Model:** `google/flan-t5-base` fine-tuned on the `databricks/dolly-15k` instruction-following dataset.
* **Quantization:** Tensors are loaded utilizing `BitsAndBytesConfig` in strict 8-bit precision (`load_in_8bit=True`), preventing immediate GPU VRAM fragmentation during context generation.
* **Adapter Integration:** The base model is merged with a Low-Rank Adaptation (LoRA) via `peft`, allowing task-specific instruction alignment without the computational overhead of full-parameter fine-tuning.
* **The ASGI Gateway:** The generation logic is wrapped in a FastAPI asynchronous event loop.

### Phase 2: The Security Perimeter (State Layer)
Standard localized LLM nodes fracture under concurrent payloads due to in-memory dictionaries. This stack moves that state out of process memory into persistent, decoupled engines.
* **Authorization (PostgreSQL):** Hardcoded credentials are fundamentally insecure. API keys are validated against a persistent PostgreSQL volume utilizing `asyncpg` and SQLAlchemy V2, ensuring non-blocking database I/O during the FastAPI lifespan.
* **Atomic Rate Limiting (Redis):** Transient token buckets fail under concurrency. We enforce a localized Redis container executing asynchronous Lua pipelines (`transaction=True`). This guarantees atomic evaluations of payload frequency, aggressively returning HTTP 429s to hostile actors before the requests can penetrate the GPU inference queue.

### Phase 3: Internal Orchestration (Routing Layer)
Host operating systems introduce uncontrollable port collisions. The application layer is entirely severed from the localized host environment.
* **Zero-Trust Bridge:** All containers operate strictly within an internal Docker bridge (`edge-network`). 
* **The Ingress Tunnel:** A Cloudflare `cloudflared` daemon negotiates a direct HTTP2 TCP tunnel to the global perimeter, avoiding inbound port forwarding and local firewall ACLs.
* **Internal Reverse Proxy:** Trans-continental payloads pierce the tunnel and are intercepted by Traefik. Traefik dynamically routes the HTTP traffic to the Uvicorn workers entirely within the isolated bridge, leaving zero ports exposed to the host machine.

### Phase 4 & 5: Observability and Chaos Engineering
A functional inference node without telemetry is an operational black box. This matrix integrates a strict observability layer to monitor the perimeter defense.
* **The TSDB Scraper:** Prometheus silently scrapes the Uvicorn workers every 5 seconds. The `/metrics` endpoint is strictly whitelisted from the Redis token bucket to prevent a self-inflicted denial of service on the telemetry layer.
* **Declarative Dashboards:** Grafana is provisioned via Infrastructure as Code (IaC). Dashboards are etched directly into the container state, requiring zero manual UI configuration.

#### Benchmarks: The Death of Transient Memory
To see how the stack behaves under concurrency, a 150-concurrent-user Locust swarm was run against the Cloudflare tunnel.

<img width="916" height="335" alt="429 RPS Log" src="https://github.com/user-attachments/assets/9c23d47a-ae0a-46f9-a664-cd94f2234368" />

> System state during a 150-concurrent-user synthetic load test. Redis pipelines returning HTTP 429 to requests over the bucket limit, so the inference queue only sees accepted traffic. Single machine, one run — the screenshot is the whole evidence base, and no latency percentile was recorded.

---

## Deployment Protocol

### 1. Production Matrix Compilation
To compile and boot the isolated infrastructure:
```bash
docker-compose up -d --build
```
The PostgreSQL schema will initialize, the Redis token bucket will arm, and Traefik will establish the internal DNS routing.

### 2. The Cloudflare Ingress Extraction
Extract the generated tunnel URL to route traffic to your local machine:

```bash
docker logs edge-ingress-tunnel
```
Locate the *.trycloudflare.com URL. All external payloads must be directed to https://<URL>/generate/.

### 3. Replicating the Chaos Engineering
The load-testing dependencies are kept out of the build context to keep the image small. To reproduce the test locally:

Install the development dependencies on your host machine:
```bash
pip install -r requirements-dev.txt
```
Run the load test against your active Cloudflare tunnel:
```bash
locust -f benchmarks/locustfile.py --host=https://<your-cloudflare-url>.trycloudflare.com
```
Open the local Grafana instance at `http://localhost:3000` to watch the Redis token bucket throttling the overflow. The admin credentials come from `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` in `docker-compose.yml` — set your own before exposing Grafana anywhere.
