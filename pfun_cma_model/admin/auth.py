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
from pfun_cma_model.admin.core import (
    Session,
    pwd_context,
    ACCESS_TOKEN_EXPIRE_MINUTES,
)
from pfun_cma_model.admin.models import User


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
        # create an access token, store in session
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        session_token = create_access_token(
            data={"usn": username, "category": category},
            expires_delta=access_token_expires,
        )
        request.session.update({"token": session_token, "uid": user.id})
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        # Check if the session contains our token
        token = request.session.get("token")
        if not token:
            return False

        # Grab the matching user from the session
        async with Session() as db_session:  # type: ignore
            result = await db_session.execute(
                select(User).where(User.id == request.session.get("uid"))
            )
            user = result.scalars().first()
            if not user:
                return False

        # Verify the decoded token contains expected username(or email), plus user category
        decoded_token = jwt.decode(
            token, key=get_settings().secret_key, algorithms=[ALGORITHM]
        )
        usn = decoded_token.get("usn")
        if not ( usn in (user.name, user.email) ):
            return False
        user_category = "admin" if user.is_admin else "user"
        if not (decoded_token.get("category") == user_category):
            return False

        return True


authentication_backend = AdminAuth(secret_key=get_settings().secret_key)
