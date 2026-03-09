import time
import redis.asyncio as redis

class DistributedRateLimiter:
    def __init__(self, requests_per_minute: int, redis_url: str = "redis://redis:6379/0") -> None:
        self.rate_limit = requests_per_minute
        self.redis = redis.from_url(redis_url, decode_responses=True)
        
    async def is_allowed(self, client_ip: str) -> bool:
        current_time = int(time.time())
        window_start = current_time - 60
        key = f"rate_limit:{client_ip}"
        
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, window_start) # Delete requests from the key olde rthan 60s, returns int requests deleted
            pipe.zcard(key) # Count requests within the new 60sec frame, returns int requests present
            pipe.zadd(key, {str(current_time): current_time}) # Logs current time, always return 1
            pipe.expire(key, 60) # Expire after 60s, completing a Rate-limit cycle, returns 0/1
        
            results = await pipe.execute() # Returns a list of 'returns' from 4 functions in the pipeline
            
        request_count = results[1] # Select value returned from z.card
        
        if request_count >= self.rate_limit:
            return False
        return True
    
limiter = DistributedRateLimiter(requests_per_minute=10)