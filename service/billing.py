import logging
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.db.config import get_db
from models.api_key import ApiKey


logger = logging.getLogger(__name__)

async def replenish_token(customer_id: str, amount_paid_cents: int, db: AsyncSession = Depends(get_db)):
    #Conversion for 1 dollar equals 1 million tokens
    tokens_replenished = (amount_paid_cents // 100) * 1000000
    
    try:
        stmt = select(ApiKey).where(ApiKey.user_id == customer_id, ApiKey.is_active == True)
        query = await db.execute(stmt)
        api_key = query.scalar_one_or_none()
        
        if api_key:
            api_key.token_balance += tokens_replenished
            await db.commit()
            logger.info(msg=f"Ledger replenished. Customer {customer_id} credited {tokens_replenished} tokens.")
        else:
            logger.error(msg=f"MoR invoice paid, but no localized API key exists for user {customer_id}.")
    
    except Exception as e:
        logger.error(msg=f"Database connection failed during token repletion: {e}")
        await db.rollback()