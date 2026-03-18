from fastapi import APIRouter, HTTPException, Request, Depends
from starlette import status
import hmac
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from core.settings import settings
from core.db.config import get_db
from service.billing import replenish_token


router = APIRouter(prefix="/lemonsqueezy", tags=["lemon"])
logger = logging.getLogger(__name__)
LEMON_SQUEEZY_WEBHOOK_SECRET = settings.LEMON_SQUEEZY_WEBHOOK_SECRET

@router.post("/")
async def create_lsqueeze_transaction(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    header = request.headers.get("x-signature")
    
    if not header or not LEMON_SQUEEZY_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cryptographic perimeter failure.")

    secret_bytes = bytes(LEMON_SQUEEZY_WEBHOOK_SECRET, 'utf-8')
    expected_signature = hmac.new(secret_bytes, payload, hashlib.sha256).hexdigest()
    
    if not hmac.compare_digest(expected_signature, header):
        logger.error("Cryptographic signature spoofing detected. Connection dropped.")
        raise HTTPException(status_code=400, detail="Invalid signature.")
    
    try:
        data = await request.json()
        event_name = data['meta']['event_name']
        custom_data = data['meta']['custom_data']
        
        if event_name == 'order_created':
            amount_paid_cents = data['data']['attributes']['total']
            user_id = custom_data.get('user_id')
            
            if user_id:
                await replenish_token(customer_id=user_id, amount_paid_cents=amount_paid_cents, db=db)
                
    except KeyError as e:
        logger.error(msg=f"Malformed MoR payload structure: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload value.")
       
    return {"status": "success"}