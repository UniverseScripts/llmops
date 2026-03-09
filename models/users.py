from core.db.config import Base
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__="users"
    
    id = Column(Integer, primary_key=True, index=True)
    
    api = relationship("ApiKey", back_populates="user")
    