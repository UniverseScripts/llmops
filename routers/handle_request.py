from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from schemas.generate import GenerateContext, GenerateResponse
import torch
import starlette.concurrency as concurrency
from service.auth import verify_api_key
from service.billing import report_token_consumption

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
    
@router.post("/", response_model=GenerateResponse, dependencies=[Depends(verify_api_key)])
async def GenerateRequest(payload: GenerateContext, request: Request, verify_api_key = Depends(verify_api_key), background_tasks = BackgroundTasks):

    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    
    if verify_api_key.token_balance <= 0:
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
        
        verify_api_key.token_balance -= tokens_consumed
        
        if verify_api_key.owner.stripe_customer_id:
            background_tasks.add_task(
                report_token_consumption,
                verify_api_key.owners.stripe_customer_id,
                tokens_consumed
            )
        
        return GenerateResponse(completion=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))