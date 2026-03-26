import logging
import os
from typing import Any
from datetime import timedelta, datetime, timezone
from dataclasses import dataclass
from sqlalchemy import select
import jwt
from passlib.context import CryptContext
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    AsyncEngine,
)
from fastapi import Security, HTTPException
from fastapi.security import APIKeyCookie
from fastapi_sso.sso.base import OpenID
from pfun_common.settings import get_settings
from pfun_cma_model.misc.pathdefs import PFunDataPaths


@dataclass
class CryptContextDefaults:
    """Default settings for the password hashing context."""

    schemes = ["bcrypt"]
    deprecated = "auto"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30


def setup_admin_backend() -> tuple[Any, AsyncEngine, Any]:
    """Setup the admin database and sqlalchemy engine."""
    Base = declarative_base()
    pfun_dpaths = PFunDataPaths()
    pfun_dpaths.ensure_local_share_path_exists()
    engine = create_async_engine(
        pfun_dpaths.admin_db_fpath,
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine, class_=AsyncSession)  # type: ignore
    return Base, engine, Session


# Initialize the admin backend (database, engine, session maker)
Base, engine, Session = setup_admin_backend()


def setup_pwd_context() -> CryptContext:
    """Setup the password context for hashing and verifying passwords."""

    # Initialize password context for hashing and verifying passwords
    local_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    local_pwd_context.load_path(
        os.path.join(PFunDataPaths._pfun_root_path, "SECURITY_POLICY.ini")
    )
    return local_pwd_context


# Initialize the password context for hashing and verifying passwords
pwd_context = setup_pwd_context()


# --- Auth core methods: ---


async def get_user(db, username: str) -> None | Any:
    """Retrieve the user from the database."""
    from pfun_cma_model.admin.models import User

    user = None
    async with Session() as db_session:  # type: ignore
        result = await db_session.execute(
            select(User).where((User.name == username) | (User.email == username))
        )
        user = result.scalars().first()
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """Create a JWT access token with the provided data and expiration time."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=CryptContextDefaults.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, get_settings().secret_key, algorithm=CryptContextDefaults.ALGORITHM
    )
    return encoded_jwt


def verify_password(plain_password, hashed_password):
    """Verify the provided plain password against the hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash the password using the password context."""
    return pwd_context.hash(password)


async def authenticate_user(db, username: str, password: str):
    """Authenticate the user by verifying the provided username and password against the database."""
    user = await get_user(db, username)
    if not user:
        verify_password(
            password, "notavalidpasswordatall"
        )  # to update the state of the crypt context
        return False
    if not verify_password(password, user.hashed_password):
        logging.debug("Failed login attempt for username/email: %s", username)
        return False
    return user


from typing import Annotated
from fastapi import Depends


async def get_logged_user(
    cookie: str | bytes = Security(APIKeyCookie(name="token")),
) -> OpenID:
    """Get user's JWT stored in cookie 'token', parse it and return the user's OpenID.

    This function can be used as a dependency in your admin views to get the **currently logged-in user.**

    NOTE: (for authorization) Used by fastapi-sso to get the logged-in user from the JWT token stored in the cookie.
        The JWT token is created in the `login` method of the `AdminAuth` class in `auth.py`.
    """
    try:
        claims = jwt.decode(
            cookie,
            key=get_settings().secret_key,
            algorithms=[CryptContextDefaults.ALGORITHM],
        )
        return OpenID(**claims["pld"])
    except Exception as error:
        raise HTTPException(
            status_code=401, detail="Invalid authentication credentials"
        ) from error


OIDAuthenticatedUser = Annotated[OpenID, Depends(get_logged_user)]
#: Authenticated User (OpenID credentials)
