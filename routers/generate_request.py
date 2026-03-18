from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.generate import GenerateContext, GenerateResponse
import torch
import starlette.concurrency as concurrency
from service.verify_api import verify_api_key
from core.db.config import get_db


router = APIRouter(prefix="/generate", tags=["generate"])

def synchronous_generation(prompt: str, model, tokenizer, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).to(model.device)
        
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
@router.post("/", response_model=GenerateResponse)
async def GenerateRequest(payload: GenerateContext, request: Request, api_key = Depends(verify_api_key), db: AsyncSession = Depends(get_db)):

    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    
    if api_key.token_balance <= 0:
        raise HTTPException(status_code=402, detail="Payment Required: Token balance exhausted.")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model uninitialized in VRAM")
    
    prompt = f"Instruction: {payload.instructions}\n Context: {payload.context}\n Response:"
    
    try:
        result = await concurrency.run_in_threadpool(
            synchronous_generation,
            prompt,
            model,
            tokenizer,
            payload.max_new_tokens,
        )
        
        tokens_consumed = len(prompt.split()) + len(result.split())
        
        api_key.token_balance -= tokens_consumed
        await db.commit()

        return GenerateResponse(completion=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))