from typing import Optional
import asyncio
from sqlalchemy import Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .core import Base, engine

__all__ = ["User", "Site"]


# --- Define Admin Models ---


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    email: Mapped[str] = mapped_column(String, nullable=False, index=True, unique=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    site_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("sites.id"), nullable=True, default=None
    )
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    bio: Mapped[str] = mapped_column(String, nullable=True)
    site: Mapped[Optional["Site"]] = relationship(back_populates="users")
    hashed_password: Mapped[str] = mapped_column(String)


class Site(Base):
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    users: Mapped[list["User"]] = relationship(back_populates="site")


async def init_models():
    """Initialize the database models (create tables)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)  # Drop existing tables
        await conn.run_sync(Base.metadata.create_all)  # Create tables
