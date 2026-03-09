from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from core.db.config import get_db
from models.api_key import ApiKey

API_KEY_NAME = "X-Enterprise-Token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header), db: AsyncSession = Depends(get_db)):
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key")
    
    stmt = select(ApiKey).where(ApiKey.valid_api_keys == api_key)
    query = await db.execute(stmt)
    db_key = query.scalar_one_or_none()
    
    if not db_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    
    return db_key
