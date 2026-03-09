from core.db.config import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from models.users import users

class ApiKey(Base):
    __tablename__="api_key"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey(users.id))
    
    valid_api_keys = Column(String, nullable=False)
    
    users = relationship(users.__tablename__, back_populates=users.api)
    
    
api_key = ApiKey()