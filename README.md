# Enterprise Distributed Edge Inference Cluster (Local LLMOps)

A hardened, fully distributed Dockerized edge-inference pipeline. It routes trans-continental B2B API traffic to highly constrained local hardware (≤ 4GB VRAM) utilizing stateful token-bucket algorithms and persistent authorization, achieving zero cloud compute costs.

This repository bypasses the financial overhead of hyperscaler GPU instances and managed cloud caching (AWS ElastiCache/RDS) by orchestrating a localized quantized LLM matrix alongside a Zero Trust ingress tunnel, Traefik reverse proxy, Redis, and PostgreSQL.

## Architectural Features

1. **Atomic Distributed Rate Limiting:** Eradicates transient ASGI memory. Executes asynchronous Lua pipelines against a localized Redis container (`transaction=True`) to enforce strict hardware boundaries and prevent token-bucket race conditions under concurrent worker loads.
2. **Persistent Security Perimeter:** Replaces hardcoded authorization sets with an asynchronous PostgreSQL session (`asyncpg` + SQLAlchemy V2), authenticating global edge requests natively without blocking the ASGI event loop.
3. **Internal Reverse Proxy & Load Balancing:** Orchestrates a Traefik load balancer to intercept inbound payloads from the Cloudflare Quick Tunnel, seamlessly distributing traffic across the internal FastAPI inference engine.
4. **Strict State Isolation:** Implements 8-bit quantization (`bitsandbytes`) for `google/flan-t5-base`. Precision LoRA weights are mapped strictly to the `app.state` on boot, preventing memory fragmentation.

## Prerequisites

To execute this orchestration matrix, the host machine must possess:
* Docker Engine & Docker Compose (v3.8+)
* NVIDIA Container Toolkit (for GPU passthrough)
* A CUDA-compatible GPU

## Deployment Matrix

**1. Clone the repository:**
```bash
git clone https://github.com/universescripts/llmops.git
cd llmops
```
**2. Establish the Environment Schema:**
Create a .env file in the root directory to define the database initialization parameters:
```Plaintext
PGUSER=postgres
PGPASS=enterprise_secure_password
PGDB=llmops
PGHOST=edge-db
PGPORT=5432
```
**3. Boot the Orchestration (Detached):**
```bash
docker-compose up -d --build
```
**4. The Security Seeding Protocol:**
Wait 15 seconds for the database engine to initialize its superuser and the FastAPI lifespan to generate the SQLAlchemy schemas. Then, inject the foundational administrative records:

```bash
docker exec -it edge-db psql -U postgres -d llmops -c "
INSERT INTO users (id) VALUES (1);
INSERT INTO api_key (id, user_id, valid_api_keys) VALUES ('sk_live_edge_node_001', 1, 'sk_live_edge_node_001');
"
```
**5. Extract the Global Endpoint:**
The Cloudflare Quick Tunnel dynamically generates a secure HTTPS endpoint routing to Traefik. Extract it from the daemon logs:

```bash
docker logs edge-ingress-tunnel
```
(Look for the URL ending in *.trycloudflare.com)

## Telemetry & API Usage Protocol
The endpoint strictly requires the trailing slash and a valid Enterprise Token header verified against the PostgreSQL volume.

**The Global Strike (cURL):**
```bash
curl -X POST "https://<your-generated-url>[.trycloudflare.com/generate/](https://.trycloudflare.com/generate/)" \
     -H "Content-Type: application/json" \
     -H "X-Enterprise-Token: sk_live_edge_node_001" \
     -d '{"prompt": "Explain the concept of active nihilism in software architecture."}'
```
**The Infrastructure Verification:**
To verify the distributed rate limiter is actively defending the edge hardware without race conditions, monitor the atomic Redis pipelines in real-time during an active request:
```bash
docker exec -it edge-db-redis redis-cli monitor
```
## Phase 3: Production Scaling
The Phase 3 architecture resolves the transient state fracturing anomalies of standard localized edge nodes. For further deployment into a bare-metal multi-node physical cluster (e.g., Kubernetes/Docker Swarm), the local PostgreSQL volume must be decoupled into a highly available cluster, and Traefik must be configured for distributed swarm routing.

## Phase 4 & 5: Observability & Chaos Engineering
A functional inference node without telemetry is an operational black box. This matrix integrates a strict Zero-Trust observability layer bound entirely within the internal Docker bridge, bypassing host network exposure while capturing real-time trans-continental ingress metrics.

### The Telemetry Matrix
* **Prometheus:** A Time Series Database (TSDB) silently scraping the Uvicorn workers every 5 seconds, isolated from the Redis rate-limiting perimeter to prevent self-inflicted denial of service.
* **Grafana:** Declaratively provisioned via Infrastructure as Code (IaC). Dashboards are etched into the container state, mapping atomic API requests and 95th percentile (p95) ASGI event loop latency.

### Benchmarks: The Death of Transient Memory
To mathematically prove the architecture's load-bearing capability, the node was subjected to chaos engineering. A 150-concurrent-user synthetic swarm was deployed against the Cloudflare Zero-Trust tunnel.

*(Insert Grafana Dashboard Screenshots Here)*
*System state during a 150-concurrent-user synthetic load test. Redis asynchronous pipelines actively throttling trans-continental overflow (HTTP 429) to preserve ASGI event loop integrity and maintain stable p95 latency for accepted payloads.*

### Replicating the Chaos Engineering
The load-testing matrix is strictly segregated from the production build context to maintain minimal image size and eliminate CVE vulnerabilities. To verify the B2B edge ingress resilience locally:

1. Install the development dependencies on your host machine:
   ```bash
   pip install -r requirements-dev.txt
   ```
Execute the trans-continental swarm against your active Cloudflare tunnel:

```bash
locust -f benchmarks/locustfile.py --host=https://<your-cloudflare-url>.trycloudflare.com
```
Navigate to the local Grafana instance (http://localhost:3000) utilizing the orchestrated administrative credentials to monitor the Redis token bucket throttling the overflow in real-time.
