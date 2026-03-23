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
    """Custom authentication backend for sqladmin using SSO credentials."""

    async def login(self, request: Request) -> bool:
        return True

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        request.session.update({"token": None})
        request.cookies.pop("token", None)
        return True

    async def authenticate(self, request: Request) -> bool:
        logging.debug(
            "Checking for SSO-based authentication. Session data: %s",
            str(request.session),
        )

        # We look for the token either in the session or in cookies
        # because the sso route stores it in both.
        cookie = request.cookies.get("token")
        session_token = request.session.get("token")

        token = cookie or session_token

        if token is None:
            logging.warning(
                "No authentication token found. Session data: %s",
                str(request.session),
            )
            return False

        try:
            decoded_token = jwt.decode(
                token, key=get_settings().secret_key, algorithms=[CryptContextDefaults.ALGORITHM]
            )
            logging.debug("Decoded JWT token for SSO login: %s", decoded_token)

            # The token format depends on where it comes from:
            # - If it's directly from our SSO flow, it might have "pld" with openid info.
            # - If it's the token we re-issued (if we kept that logic), it might have "usn" and "category".

            email = None
            openid_info = None
            if "pld" in decoded_token:
                openid_info = OpenID(**decoded_token["pld"])
                email = openid_info.email
            elif "usn" in decoded_token:
                email = decoded_token["usn"]
            else:
                logging.warning(
                    "JWT token does not contain expected 'pld' or 'usn' claims. Token: %s, Session data: %s",
                    token,
                    str(request.session),
                )
                return False

        except InvalidTokenError as error:
            logging.error(
                "Invalid authentication token.\n\t+ Token: %s\n\t+ Session data: %s.\n\t+ Error: %s",
                token,
                str(request.session),
                str(error),
            )
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            ) from error
        except Exception as error:
            logging.error(
                "Error occurred while decoding JWT token.\n\t+ Token: %s\n\t+ Session data: %s.\n\t+ Error: %s",
                token,
                str(request.session),
                type(error).__name__ + ": " + str(error),
            )
            raise HTTPException(
                status_code=401, detail="Invalid authentication credentials"
            )

        async with Session() as db_session:  # type: ignore
            # Look up the user by email, since it's the primary key for SSO
            result = await db_session.execute(
                select(User).where(User.email == email)
            )
            user = result.scalars().first()
            if not user:
                logging.warning(
                    f"Login attempt with non-existent email: {email}"
                )
                if get_settings().debug:
                    logging.info("Debug mode enabled. Automatically creating user for %s", email)
                    display_name = openid_info.display_name if openid_info else email
                    user = User(
                        email=email,
                        name=display_name or email,
                        hashed_password="SSO_CREATED",
                        is_admin=False,
                        age=None,
                        bio=None,
                        site_id=None
                    )
                    db_session.add(user)
                    await db_session.commit()
                    await db_session.refresh(user)
                else:
                    return False  # User not found

            # Update session with uid for further requests
            request.session.update({"uid": user.id})

        return True


authentication_backend = AdminAuth(secret_key=get_settings().secret_key)
