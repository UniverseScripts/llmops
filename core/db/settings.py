from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PGUSER: str
    PGPASS: str
    PGHOST: str
    PGDB: str
    PGPORT: int
    
    @property
    def DATABASE_URL(self):
        from urllib.parse import quote_plus
        
        user = quote_plus(self.PGUSER)
        password = quote_plus(self.PGPASS)
    
        return f'postresql+asyncpg://{user}:{password}@{self.PGHOST}:{self.PGPORT}/{self.PGDB}'
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
            
            
settings = Settings()