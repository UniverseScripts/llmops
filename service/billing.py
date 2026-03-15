import stripe
from stripe import StripeError
import os
import logging
from fastapi import HTTPException

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
        stripe.billing.MeterEvent.create(
            event_name="llm_inference_tokens",
            payload={
                "value": str(tokens_consumed),
                "stripe_customer_id": stripe_customer_id
            }
        )
    except StripeError as e:
        logger.error(f"Failed to report {tokens_consumed} tokens for {stripe_customer_id}: str{e}")
    