from pydantic import BaseModel, Field, EmailStr

class UserCreate(BaseModel):
    username: str = Field(..., description="Contains username for registration.")
    email: EmailStr = Field(..., description="Contains user's email for registration.")
    password_input: str = Field(..., description="Contains user's password input for hashing.")
    
class UserResponse(BaseModel):
    username: str = Field(..., description="Contains username for response.")
    email: EmailStr = Field(..., description="Contains user's email for response.")
    message: str = Field(default="", description="Contains status code to notify whether registration was successful or not.")
    
class AccessToken(BaseModel):
    access_token: str
    token_type: str