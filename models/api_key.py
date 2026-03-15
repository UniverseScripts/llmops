from core.db.config import Base
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean
from sqlalchemy.orm import relationship

class ApiKey(Base):
    __tablename__="api_key"
    #Inference Tier
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    valid_api_keys = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False)
    
    tier = Column(String, default="Developer")
    token_balance = Column(Integer, default=1000000)
    
    user = relationship("User", back_populates="api")