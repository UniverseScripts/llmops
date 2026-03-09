from core.db.config import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class ApiKey(Base):
    __tablename__="api_key"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    valid_api_keys = Column(String, nullable=False)
    
    user = relationship("User", back_populates="api")
    
    
api_key = ApiKey()