import logging

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
from datetime import timedelta
from typing import Annotated, Optional
from fastapi_sso import OpenID
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyCookie
from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from sqlalchemy import select
import jwt
from jwt.exceptions import InvalidTokenError
from pfun_common.settings import get_settings
from pfun_cma_model.admin.core import (
    Session,
    get_logged_user,
    pwd_context,
    create_access_token,
    CryptContextDefaults,
)
from pfun_cma_model.admin.models import User

logger.setLevel(level=logging.DEBUG if get_settings().debug is True else logging.INFO)


class AdminAuth(AuthenticationBackend):
    """Custom authentication backend for sqladmin using username/password credentials."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        category = "admin" if form.get("is_admin") else "user"
        openid_info = None
        if any([not username, not password]):
            # If username or password is not provided, check if this is an SSO-based login attempt (e.g. Google SSO) by looking for the JWT token in the session
            # Check instead for SSO-based authentication (e.g. Google SSO) by looking for the JWT token in the session
            logging.debug(
                "Username or password not provided, checking for SSO-based authentication. Session data: %s",
                str(request.session),
            )
            cookie = request.cookies.get("token")
            if cookie is None:
                logging.warning(
                    "No authentication token found in session. Session data: %s",
                    str(request.session),
                )
                return False
            try:
                decoded_token = jwt.decode(
                    cookie, key=get_settings().secret_key, algorithms=[CryptContextDefaults.ALGORITHM]
                )
                logging.debug("Decoded JWT token for SSO login: %s", decoded_token)
                if "pld" not in decoded_token:
                    logging.warning(
                        "JWT token does not contain expected 'pld' claim. Token: %s, Session data: %s",
                        cookie,
                        str(request.session),
                    )
                    return False
                openid_info = OpenID(**decoded_token["pld"])
                user = (
                    openid_info.email
                )  # Assuming email is used as the username for SSO users
                
            except InvalidTokenError as error:
                logging.error(
                    "Invalid authentication token.\n\t+ Token: %s\n\t+ Session data: %s.\n\t+ Error: %s",
                    cookie,
                    str(request.session),
                    str(error),
                )
                raise HTTPException(
                    status_code=401, detail="Invalid authentication credentials"
                ) from error
            except Exception as error:
                logging.error(
                    "Error occurred while decoding JWT token.\n\t+ Token: %s\n\t+ Session data: %s.\n\t+ Error: %s",
                    cookie,
                    str(request.session),
                    type(error).__name__ + ": " + str(error),
                )
                raise HTTPException(
                    status_code=401, detail="Invalid authentication credentials"
                )
        elif all([username is not None, password is not None]):
            # Validate username/password credentials
            async with Session() as db_session:  # type: ignore
                result = await db_session.execute(
                    select(User).where(
                        (User.name == username) | (User.email == username)
                    )
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
                if not ok:
                    logging.warning(
                        "Invalid credentials (user=%s, ok=%s)", str(user), str(ok)
                    )
                    return False  # Invalid credentials
                # If the hash needs to be updated (e.g. if the hashing algorithm has changed), update it in the database
                if new_hash:
                    user.hashed_password = new_hash
                    await db_session.commit()

        # Successful login, update session with user info and token
        # Update session
        # create an access token, store in session
        access_token_expires = timedelta(minutes=CryptContextDefaults.ACCESS_TOKEN_EXPIRE_MINUTES)
        session_token = create_access_token(
            data={"usn": username, "category": category},
            expires_delta=access_token_expires,
        )
        request.session.update({"token": session_token})
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        request.session.update({"token": None})
        request.cookies.pop("token", None)
        return True

    async def authenticate(self, request: Request) -> bool:
        # Check if the session contains our token
        token = request.session.get("token")
        if not token:
            logging.debug(
                "No authentication token found in session. Session data: %s",
                str(request.session),
            )
            return False

        # Grab the matching user from the session
        async with Session() as db_session:  # type: ignore
            result = await db_session.execute(
                select(User).where(User.id == request.session.get("uid"))
            )
            user = result.scalars().first()
            if not user:
                logging.warning(
                    "User not found in session. Session data: %s", str(request.session)
                )
                return False

        # Verify the decoded token contains expected username(or email), plus user category
        decoded_token = jwt.decode(
            token,
            key=get_settings().secret_key,
            algorithms=[CryptContextDefaults.ALGORITHM],
        )
        usn = decoded_token.get("usn")
        if not (usn in (user.name, user.email)):
            logging.warning(
                "Username/email in token does not match user in session. Token usn: %s, User name: %s, User email: %s",
                usn,
                user.name,
                user.email,
            )
            return False
        user_category = "admin" if user.is_admin else "user"
        if not (decoded_token.get("category") == user_category):
            logging.warning(
                "User category mismatch in token. Expected: %s, Found: %s",
                user_category,
                decoded_token.get("category"),
            )
            return False

        return True


authentication_backend = AdminAuth(secret_key=get_settings().secret_key)
