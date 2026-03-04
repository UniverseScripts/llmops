from fastapi import FastAPI, Request
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
from contextlib import asynccontextmanager
from routers import handle_request
import time
import logging


@asynccontextmanager
async def Lifespan(app: FastAPI):
    print("🚀 Starting Enterprise Inference Node...")
    
    model_id="google/flan-t5-base"
    peft_model_dir="./core/lora-flan-t5-dolly"
    
    try:
        print("🔺Initializing base Flan-T5 Model Tensors...")
        
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        print("🔻Injecting LoRA-trained Model Tensors...")
        peft_model = PeftModel.from_pretrained(base_model, peft_model_dir)
        peft_model.eval()
        
        app.state.model = peft_model
        app.state.tokenizer = tokenizer
        print("✔️ Injection Complete! Inference Node active!")
        
    except Exception as e:
        print(f"✖️ Failed to setup Inference Node. Reason: {e}")
        raise e
    
    yield
    print("💣 Shutting down Enterprise Inference Node...")
    del app.state.model
    del app.state.tokenizer
    torch.cuda.empty_cache()

app = FastAPI(title="Enterprise-grade Inference API", description="A backend router for scalable inference", version="0.1.0", lifespan=Lifespan)

@app.get("/", description="Perform Health check on the Inference Node.")
async def HealthCheck():
    return {"health": "healthy", "status": "active"}
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMOps Telemetry")

@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(
        f"Method: {request.method} | Path: {request.url.path} | "
        f"Status: {response.status_code} | Latency: {process_time:.4f}s"
    )

    return response

app.include_router(handle_request.router)