# Enterprise Edge Cluster: Distributed Local LLMOps (Stripe Billing Integration)
## The branch "enterprise-saas" integrates Stripe, including setups such as account provision and token billing.
## Status: Finished, branch frozen.

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

### Phase 6: The Enterprise Monetization Matrix (SaaS Edition)
The open-source inference engine is fundamentally unmetered. This `enterprise-saas` branch introduces a rigorous, stateful financial perimeter designed to extract B2B capital without fracturing the core inference latency. 
The architecture achieves this by strictly bifurcating identity authorization from capital metering.
 
1. The API Gateway & Stateless Authorization
To prevent database I/O bottlenecks during concurrent inference, human identity and programmatic access are structurally isolated.
- **Human Dashboard (Stateless):** Enterprise administrators authenticate via standard credentials. The node issues a JSON Web Token (JWT). The cryptography mathematically proves identity, allowing the UI to interact with the billing layer without querying the database on every render.
- **Programmatic Access (Stateful):** Human users provision deterministic, localized API keys (`sk_live_...`). These keys are bound directly to the PostgreSQL database, acting as the bridge between the inference request and the global capital ledger.

2. The Asynchronous Financial Ledger (Stripe)
Flat-rate billing mathematically guarantees infrastructure ruin under LLM compute loads. This matrix strictly enforces consumption-based token metering.
- **Local Deduction:** The Redis token bucket intercepts the ASGI event loop, instantly calculating the tokenizer length and deducting the payload from the localized token balance.
- **Global Synchronization:** To preserve the p95 latency of the HTTP response, the trans-continental network call to the Stripe Metered Billing API is deferred to a FastAPI `BackgroundTask`. The node extracts capital asynchronously.

3. The Cryptographic Webhook Perimeter
Client-side financial state is inherently invalid. Capital replenishment is handled exclusively via server-to-server communication.
- Inbound Stripe payloads signaling successful invoice payments are intercepted by `routers/webhooks.py`.
- The payload is verified against a localized symmetric cryptographic signature (`STRIPE_WEBHOOK_SECRET`). Spoofed payloads are violently dropped. Authenticated payloads directly execute an asynchronous `db.commit()` to refill the enterprise client's token ledger.

#### Benchmarks: The Death of Transient Memory
To mathematically prove the architecture's load-bearing capability, the node was subjected to chaos engineering. A 150-concurrent-user synthetic swarm was deployed against the Cloudflare Zero-Trust tunnel.

<img width="916" height="335" alt="429 RPS Log" src="https://github.com/user-attachments/assets/9c23d47a-ae0a-46f9-a664-cd94f2234368" />

> System state during a 150-concurrent-user synthetic load test. Redis asynchronous pipelines actively throttling trans-continental overflow (HTTP 429) to preserve ASGI event loop integrity and maintain stable p95 latency for accepted payloads.*

---

## Deployment Protocol

### 1. Environmental Provisioning
Create an `.env` file in the root directory mapping your Stripe developer credentials:
```env
PGUSER=postgres
PGPASSWORD=placeholder_enter_your_psql_pw_here
PGDATABASE=llmops
PGHOST=localhost
PGPORT=5432
SECRET_KEY=Ubdzme...
ALGORITHM=HS256
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

### 2. Production Matrix Compilation
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

### 3. Ingress and Registration
Route to your Cloudflare tunnel ingress. You must execute a POST /register payload first. The node will intercept the registration, provision a global Stripe Customer ID, and map it to your local PostgreSQL row before confirming creation.

### 4. Replicating the Chaos Engineering
The load-testing matrix is strictly segregated from the production build context to maintain minimal image size and eliminate CVE vulnerabilities. To verify the B2B edge ingress resilience locally:

Install the development dependencies on your host machine:
```bash
pip install -r requirements-dev.txt
```
Execute the trans-continental swarm against your active Cloudflare tunnel:
```bash
locust -f benchmarks/locustfile.py --host=https://<your-cloudflare-url>.trycloudflare.com
```
Open the local Grafana instance at `http://localhost:3000` to watch the Redis token bucket throttling the overflow. The admin credentials come from `GF_SECURITY_ADMIN_USER` / `GF_SECURITY_ADMIN_PASSWORD` in `docker-compose.yml` — set your own before exposing Grafana anywhere.
