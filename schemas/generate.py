from pydantic import BaseModel, Field

class GenerateContext(BaseModel):
    instructions: str = Field(..., min_length=5, description="Primary instructions for the model")
    context: str = Field(default="", description="Optional context or input data for the model")
    max_new_tokens: int = Field(default=128, le=512, description="Output length constraint")
    
class GenerateResponse(BaseModel):
    completion: str