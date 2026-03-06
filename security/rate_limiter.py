import time

class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.rate_limit = requests_per_minute
        self.clients = {}
        
    def is_allowed(self, client_ip: str) -> bool:
        current_time = time.time()
        if client_ip not in self.clients:
            self.clients[client_ip] = []
            
        self.clients[client_ip] = [t for t in self.clients[client_ip] if current_time - t < 60]
        
        if len(self.clients[client_ip]) >= self.rate_limit:
            return False
        
        self.clients[client_ip].append(current_time)
        return True
    
limiter = RateLimiter(requests_per_minute=10)