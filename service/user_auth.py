from core.settings import settings
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from datetime import timedelta
from core.db.config import get_db
from models.users import User


SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM

bcrypt_context = CryptContext(schemes=['bcrypt'])
OAuth2Bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

async def authenticate_user(user_id: str, input_password: str, db: AsyncSession = Depends(get_db)):
    
    stmt = select(User).where(User.id == user_id)
    query = await db.execute(stmt)
    user = query.scalar_one_or_none()
    
    if user is None or not user:
        return False
    
    password_encoded = input_password.encode("utf-8")[:72]
    password_truncated = password_encoded.decode("utf-8", errors="ignore")
    
    if not bcrypt_context.verify(password_truncated, str(user.hashed_password)):
        return False
    
    return user

def create_access_token(user_id: str, username: str, expire_data: timedelta | None = None) -> str:
    to_json = {"sub": username, "id": user_id}
    
    if expire_data:
        expires_by = datetime.now(timezone.utc) + expire_data
    else:
        expires_by = datetime.now(timezone.utc) + timedelta(days=1)
        
    to_json.update({"exp": expires_by})
    
    encoded = jwt.encode(to_json, key=SECRET_KEY, algorithm=ALGORITHM)
    return encoded

async def get_current_user(token: str = Depends(OAuth2Bearer), db: AsyncSession = Depends(get_db)):
    global_error = HTTPException(status_code=402, detail="Failed to authorize current user.")
    
    try:
        decoded = jwt.decode(token=token, key=SECRET_KEY, algorithms=ALGORITHM)
        user_id = decoded["id"]
        
        if not user_id or user_id is None:
            raise global_error
        
        stmt = select(User).where(User.id == user_id)
        query = await db.execute(stmt)
        user = query.scalar_one_or_none()
        
        if not user or user is None:
            raise global_error
        
        return user
    except JWTError:
        raise global_error