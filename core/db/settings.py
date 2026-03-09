from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PGUSER: str
    PGPASSWORD: str
    PGHOST: str
    PGDATABASE: str
    PGPORT: int
    
    @property
    def DATABASE_URL(self):
        from urllib.parse import quote_plus
        
        user = quote_plus(self.PGUSER)
        password = quote_plus(self.PGPASSWORD)
    
        return f'postgresql+asyncpg://{user}:{password}@{self.PGHOST}:{self.PGPORT}/{self.PGDATABASE}'
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
            
            
settings = Settings()