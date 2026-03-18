from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from starlette import status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.users import User
from schemas.auth import UserCreate, UserResponse, AccessToken
from service.user_auth import create_access_token, authenticate_user, bcrypt_context
from core.db.config import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/", response_model=UserResponse)
async def sign_up(request: UserCreate, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(User.email==request.email)
    query = await db.execute(stmt)
    user = query.scalar_one_or_none()
    
    if user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User's email already exists, cannot provision account.")
    
    password_encoded = request.password_input.encode('utf-8')[:72]
    password_truncate = password_encoded.decode('utf-8', errors="ignore")
 
    create_user_model = User(
        username = request.username,
        hashed_password = bcrypt_context.hash(password_truncate),
        email = request.email,
        is_active = True,
    )
    
    db.add(create_user_model)
    try:
        await db.commit()
        await db.refresh(create_user_model)
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Database transaction failure.")
    
    return UserResponse(username=create_user_model.username, email=create_user_model.email, message="Success, user has been created!")


@router.post("/token", response_model=AccessToken)
async def login(form_data = Depends(OAuth2PasswordRequestForm), db: AsyncSession = Depends(get_db)):
    valid_user = await authenticate_user(user_id=form_data.username, input_password=form_data.password, db=db)
    if not valid_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials, authorisation failed.")
    
    valid_token = create_access_token(user_id=valid_user.id, username=valid_user.username, expire_data=None)
    
    return AccessToken(access_token=valid_token, token_type="bearer")