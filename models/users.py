from core.db.config import Base
from models.api_key import api_key
from sqlalchemy import Column, Integer
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__="users"
    
    id = Column(Integer, primary_key=True, index=True)
    
    api = relationship(api_key.__tablename__, back_populates=api_key.users)
    
users = User()