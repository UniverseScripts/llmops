from fastapi import APIRouter, HTTPException, Request, Depends
from starlette import status
import stripe
from stripe import Webhook
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from core.settings import settings
from core.db.config import get_db
from service.billing import replenish_token


router = APIRouter(prefix="/stripe", tags=["stripe"])
logger = logging.getLogger(__name__)
STRIPE_WEBHOOK_SECRET = settings.STRIPE_WEBHOOK_SECRET

@router.post("/")
async def create_stripe_transaction(request: Request, db: AsyncSession = Depends(get_db)):
    payload = await request.body()
    header = request.headers.get("stripe-signature")
    
    if not header or STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Stripe's Token Authentication failed. Try sending a valid key.")
    
    try:
        event = Webhook.construct_event(payload=payload, sig_header=header, secret=STRIPE_WEBHOOK_SECRET)
        
    except ValueError:
        logger.error(msg="Invalid Payload value.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid payload value.")
    
    except stripe.SignatureVerificationError:
        logger.error(msg="Cryptographic signature spoofing detected. Connection dropped.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature.")
    
    
    if event['type'] == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        customer_id = invoice.get('customer')
        amount_paid_cents = invoice.get('amount_paid')
        
        if customer_id: await replenish_token(stripe_customer_id=customer_id, amount_paid_cents=amount_paid_cents, db=db)
    
    elif event['type'] == 'invoice.payment_failed':
        logger.error(msg="Error occured during payment transaction.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Error occuured during transaction.")
       
    return {"status": "success"}