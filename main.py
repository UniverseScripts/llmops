from fastapi import FastAPI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch
from contextlib import asynccontextmanager
from routers import handle_request


@asynccontextmanager
async def Lifespan(app: FastAPI):
    global base_model, tokenizer
    print("🚀 Starting Enterprise Inference Node...")
    
    model_id="google/flan-t5-base"
    peft_model_dir="./core/lora-flan-t5-dolly"
    
    try:
        print("🔺Initializing base Flan-T5 Model Tensors...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_8bit=True,
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

app.include_router(handle_request.router)