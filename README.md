# Enterprise Edge Cluster: Distributed Local LLMOps

A globally routable, zero-trust edge inference node engineered to serve B2B payloads without hyperscaler cloud billing. This architecture eradicates transient memory vulnerabilities, utilizing atomic state pipelines and strict container isolation to serve quantized machine learning models under severe concurrent load.

## Architectural Matrix

The infrastructure is defined by strict boundary segregation, scaling from the localized GPU tensor allocation up to the trans-continental network ingress.

### Phase 1: The Inference Engine (Compute Layer)
The core compute layer abandons standard API wrapper logic in favor of a locally hosted, mathematically quantized machine learning matrix.
* **The Model:** `google/flan-t5-base` fine-tuned on the `databricks/dolly-15k` instruction-following dataset.
* **Quantization:** Tensors are loaded utilizing `BitsAndBytesConfig` in strict 8-bit precision (`load_in_8bit=True`), preventing immediate GPU VRAM fragmentation during context generation.
* **Adapter Integration:** The base model is merged with a Low-Rank Adaptation (LoRA) via `peft`, allowing enterprise-specific instruction alignment without the computational overhead of full-parameter fine-tuning.
* **The ASGI Gateway:** The generation logic is wrapped in a highly concurrent FastAPI asynchronous event loop.

### Phase 2: The Security Perimeter (State Layer)
Standard localized LLM nodes fracture under concurrent payloads due to in-memory dictionaries. This matrix eradicates localized memory, injecting persistent, decoupled state engines.
* **Authorization (PostgreSQL):** Hardcoded credentials are fundamentally insecure. API keys are validated against a persistent PostgreSQL volume utilizing `asyncpg` and SQLAlchemy V2, ensuring non-blocking database I/O during the FastAPI lifespan.
* **Atomic Rate Limiting (Redis):** Transient token buckets fail under concurrency. We enforce a localized Redis container executing asynchronous Lua pipelines (`transaction=True`). This guarantees atomic evaluations of payload frequency, aggressively returning HTTP 429s to hostile actors before the requests can penetrate the GPU inference queue.

### Phase 3: Internal Orchestration (Routing Layer)
Host operating systems introduce uncontrollable port collisions. The application layer is entirely severed from the localized host environment.
* **Zero-Trust Bridge:** All containers operate strictly within an internal Docker bridge (`edge-network`). 
* **The Ingress Tunnel:** A Cloudflare `cloudflared` daemon negotiates a direct HTTP2 TCP tunnel to the global perimeter, mathematically bypassing hypervisor UDP limits and local firewall ACLs.
* **Internal Reverse Proxy:** Trans-continental payloads pierce the tunnel and are intercepted by Traefik. Traefik dynamically routes the HTTP traffic to the Uvicorn workers entirely within the isolated bridge, leaving zero ports exposed to the host machine.

### Phase 4 & 5: Observability and Chaos Engineering
A functional inference node without telemetry is an operational black box. This matrix integrates a strict observability layer to monitor the perimeter defense.
* **The TSDB Scraper:** Prometheus silently scrapes the Uvicorn workers every 5 seconds. The `/metrics` endpoint is strictly whitelisted from the Redis token bucket to prevent a self-inflicted denial of service on the telemetry layer.
* **Declarative Dashboards:** Grafana is provisioned via Infrastructure as Code (IaC). Dashboards are etched directly into the container state, requiring zero manual UI configuration.

#### Benchmarks: The Death of Transient Memory
To mathematically prove the architecture's load-bearing capability, the node was subjected to chaos engineering. A 150-concurrent-user synthetic swarm was deployed against the Cloudflare Zero-Trust tunnel.

*(Insert your Grafana Phase 5 Dashboard Screenshot Here)*
> *System state during a 150-concurrent-user synthetic load test. Redis asynchronous pipelines actively throttling trans-continental overflow (HTTP 429) to preserve ASGI event loop integrity and maintain stable p95 latency for accepted payloads.*

---

## Deployment Protocol

### 1. Production Matrix Compilation
To compile and boot the isolated infrastructure:
```bash
docker-compose up -d --build
```
The PostgreSQL schema will initialize, the Redis token bucket will arm, and Traefik will establish the internal DNS routing.

### 2. The Cloudflare Ingress Extraction
Extract the dynamically generated B2B endpoint to route global traffic to your localized hardware:

```bash
docker logs edge-ingress-tunnel
```
Locate the *.trycloudflare.com URL. All external payloads must be directed to https://<URL>/generate/.

### 3. Replicating the Chaos Engineering
The load-testing matrix is strictly segregated from the production build context to maintain minimal image size and eliminate CVE vulnerabilities. To verify the B2B edge ingress resilience locally:

Install the development dependencies on your host machine:
```bash
pip install -r requirements-dev.txt
```
Execute the trans-continental swarm against your active Cloudflare tunnel:
```bash
locust -f benchmarks/locustfile.py --host=https://<your-cloudflare-url>.trycloudflare.com
```
Navigate to the local Grafana instance (`http://localhost:3000`) utilizing the orchestrated administrative credentials shown below to monitor the Redis token bucket throttling the overflow in real-time.
```plaintext
Username: admin
Password: REDACTED
```
