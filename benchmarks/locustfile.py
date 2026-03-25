from locust import HttpUser, task, between
import random

class B2BEnterpriseSwarm(HttpUser):
    # Throttle the users slightly to prevent overloading the WSL2 virtual network bridge
    wait_time = between(0.1, 0.5) 
    
    # 🔻 INJECT YOUR VALIDATED KEY HERE 🔻
    API_KEY = "sk_live_edge_node_001" 

    def _generate_spoofed_ip(self):
        # Simulates distributed global tenants to bypass the localized rate limiter
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

    @task(3)
    def transcontinental_inference_route(self):
        headers = {
            "X-Enterprise-Token": f"{self.API_KEY}",
            "X-Forwarded-For": self._generate_spoofed_ip(),
            "Content-Type": "application/json"
        }
        
        payload = {
            "instructions": "Execute enterprise financial summary.",
            "context": "Q3 Earnings report data...",
            "max_new_tokens": 100
        }
        
        # Target the headless generation endpoint
        self.client.post("/generate/", json=payload, headers=headers, name="Inference Routing")

    @task(1)
    def cluster_health_check(self):
        # Target the unprotected health endpoint to simulate baseline network polling
        self.client.get("/", name="Health Check")