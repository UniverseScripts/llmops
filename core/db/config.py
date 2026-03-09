from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from core.db.settings import settings


engine = create_async_engine(url=settings.DATABASE_URL, echo=True)

AsyncLocalSession = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    with AsyncLocalSession() as session:
        try:
            yield session
        finally:
            session.close()

           
async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)