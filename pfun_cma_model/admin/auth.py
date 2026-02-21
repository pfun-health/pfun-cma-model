import logging
from datetime import datetime, timedelta, timezone
from typing_extensions import Annotated
from fastapi import Security, Depends
from fastapi.security import APIKeyCookie
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from sqlalchemy import select
import secrets
import jwt
from jwt.exceptions import InvalidTokenError
from pfun_common.settings import get_settings
from pfun_cma_model.admin.core import Session, pwd_context
from pfun_cma_model.admin.models import User


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    request: Request, token: Depends(Annotated[str, Security(APIKeyCookie(name="token"))])
):
    """Helper function to retrieve the current user from the session."""
    user_id = request.session.get("user_id")
    if not user_id:
        return None

    # Query DB here to ensure the user hasn't been deleted
    async with Session() as db_session:  # type: ignore
        result = await db_session.execute(select(User).where(User.id == user_id))
        user = result.scalars().first()
        return user


class AdminAuth(AuthenticationBackend):
    """Custom authentication backend for sqladmin using username/password credentials."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        category = "admin" if form.get("is_admin") else "user"
        if not username or not password:
            return False

        # Validate username/password credentials
        async with Session() as db_session:  # type: ignore
            result = await db_session.execute(
                select(User).where((User.name == username) | (User.email == username))
            )
            user = result.scalars().first()
            if not user:
                logging.warning(
                    f"Login attempt with non-existent username/email: {username}"
                )
                return False  # User not found

            # Verify user exists and password is correct
            ok, new_hash = pwd_context.verify_and_update(
                str(password), user.hashed_password, category=category
            )
            if not user or not ok:
                return False  # Invalid credentials
            else:
                # If the hash needs to be updated (e.g. if the hashing algorithm has changed), update it in the database
                if new_hash:
                    user.hashed_password = new_hash
                    await db_session.commit()

        # Successful login, update session with user info and token
        # Update session
        # Generate a secure random token and store it in the session
        session_token = secrets.token_hex(16)
        request.session.update({"token": session_token, "user_id": user.id})
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        # Check if the session contains our token
        token = request.session.get("token")
        if not token:
            return False

        # Query DB here to ensure the user hasn't been deleted
        async with Session() as db_session:  # type: ignore
            result = await db_session.execute(
                select(User).where(User.id == request.session.get("user_id"))
            )
            user = result.scalars().first()
            if not user:
                return False
        return True


authentication_backend = AdminAuth(secret_key=get_settings().secret_key)
