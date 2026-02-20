from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request
from sqlalchemy import select
from passlib.context import CryptContext
import secrets
import os
from pfun_cma_model.admin.core import Session
from pfun_cma_model.admin.models import User
from pfun_cma_model.misc.pathdefs import PFunDataPaths

# Initialize password context for hashing and verifying passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
pwd_context.load_path(
    os.path.join(PFunDataPaths._pfun_root_path, "SECURITY_POLICY.ini")
)


class AdminAuth(AuthenticationBackend):
    """Custom authentication backend for sqladmin using username/password credentials."""

    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")
        if not username or not password:
            return False

        # Validate username/password credentials
        async with Session() as db_session:  # type: ignore
            result = await db_session.execute(
                select(User).where(User.username == username)
            )
            user = result.scalars().first()

            # Verify user exists and password is correct
            if not user or not pwd_context.verify_and_update(
                str(password), user.hashed_password
            ):
                return False  # Invalid credentials

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
