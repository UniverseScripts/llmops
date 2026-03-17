import stripe
from stripe import StripeError
from stripe.billing import MeterEvent
import os
import logging
from fastapi import HTTPException, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.db.config import get_db
from models.api_key import ApiKey
from models.users import User

stripe.api_key = os.getenv("STRIPE_CUSTOMER_KEY")
logger = logging.getLogger(__name__)

def provision_account(email: str) -> str:
    try:
        customer = stripe.Customer.create(email=email)
        logger.info(msg=f"Account provisioned for customer with {email}: {customer}")
        return customer.id
    
    except StripeError as e:
        logger.error(msg=f"Failed to provision an account for customer with {email}: {e}.")
        raise HTTPException(status_code=402, detail="Failed to create a Stripe Account.")


def report_token_consumption(stripe_customer_id: str, tokens_consumed: int):
    if not stripe_customer_id:
        return
    
    try:
        MeterEvent.create(
            event_name="llm_inference_tokens",
            payload={
                "value": str(tokens_consumed),
                "stripe_customer_id": stripe_customer_id
            }
        )
    except StripeError as e:
        logger.error(f"Failed to report {tokens_consumed} tokens for {stripe_customer_id}: str{e}")
        

async def replenish_token(stripe_customer_id: str, amount_paid_cents: int, db: AsyncSession = Depends(get_db)):
    #Conversion for 1 dollar equals 1 million tokens
    tokens_replenished = (amount_paid_cents // 100) * 1000000
    
    try:
        stmt = select(ApiKey).join(User).where(User.stripe_customer_id == stripe_customer_id)
        query = await db.execute(stmt)
        stripe_user = query.scalar_one_or_none()
        
        if stripe_user:
            stripe_user.token_balance += tokens_replenished
            await db.commit()
            logger.info(msg=f"Ledger replenished. Customer {stripe_customer_id} credited {tokens_replenished} tokens.")
        else:
            logger.error(msg=f"Financial ledger {stripe_customer_id} paid, but no localized API key exists.")
    
    except Exception as e:
        logger.error(msg=f"Database connection failed during token repletion: {e}")
        await db.rollback()