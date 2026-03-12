from locust import HttpUser, task, between
import json

class EnterpriseEdgeUser(HttpUser):
    # Simulate a 1 to 3 second think-time between B2B API calls
    wait_time = between(1, 3)

    def on_start(self):
        # We utilize the hardcoded key injected during the PostgreSQL seeding phase
        self.headers = {
            "Content-Type": "application/json",
            "X-Enterprise-Token": "sk_live_edge_node_001"
        }
        self.payload = {
            "prompt": "Explain the concept of active nihilism in software architecture.",
            "instructions": "Return a concise, raw system status."
        }

    @task(3)
    def generate_inference(self):
        # The primary strike against the generation endpoint
        with self.client.post("/generate/", headers=self.headers, data=json.dumps(self.payload), catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # 429 indicates our Redis atomic pipeline successfully defended the node
                response.success()
            else:
                response.failure(f"Node failure: {response.status_code}")

    @task(1)
    def health_check(self):
        # Secondary strike against the health perimeter
        self.client.get("/")