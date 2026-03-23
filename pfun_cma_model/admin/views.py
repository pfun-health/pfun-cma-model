import logging
import os
from jose import jwt
from sqladmin import BaseView, ModelView, action, expose
from sqladmin.filters import (
    BooleanFilter,
    AllUniqueStringValuesFilter,
    ForeignKeyFilter,
    OperationColumnFilter,
)
from fastapi import Request
from fastapi.responses import RedirectResponse
from sqlalchemy import func, select
from wtforms.fields import PasswordField
from wtforms.validators import InputRequired, EqualTo
from packages.pfun_common.pfun_common.settings import get_settings
from pfun_cma_model.admin.models import *
from pfun_cma_model.admin.core import CryptContextDefaults, engine, Session, pwd_context
from jose.exceptions import JWTError, ExpiredSignatureError

__all__ = ["UserAdmin", "ReportView"]

"""pfun_cma_model/admin/views.py : Admin views for pfun-cma-model."""


# Define User Admin View
class UserAdmin(ModelView, model=User):
    """Admin view for User model."""

    # Save as redirect behavior
    save_as = (
        True  # Enable "Save As" functionality to create new records from existing ones
    )
    save_as_continue = (
        False  # After "Save As", return to list view instead of edit view
    )

    # Columns configuration
    column_list = [
        User.id,
        User.name,
        User.email,
        User.is_admin,
        User.site_id,
        User.age,
    ]
    # Columns to display to the user with custom labels
    column_labels = {
        "id": "ID",
        "name": "Name",
        "email": "Email",
        "is_admin": "Is Admin",
        "site_id": "Site",
        "age": "Age",
        "bio": "Biography",
        "hashed_password": "Password",
    }
    # Filters for list view
    column_filters = [
        BooleanFilter(User.is_admin),
        AllUniqueStringValuesFilter(User.name),
        ForeignKeyFilter(User.site_id, Site.name, title="Site"),
        # OperationColumnFilter provides dropdown UI with multiple operations
        OperationColumnFilter(
            User.email
        ),  # String operations: Contains, Equals, Starts with, Ends with
        OperationColumnFilter(
            User.age
        ),  # Numeric operations: Equals, Greater than, Less than
    ]

    # Form configuration
    form_create_rules = ["name", "email", "is_admin", "age", "bio", "hashed_password"]
    form_edit_rules = ["name"]
    form_args = dict(
        hashed_password=dict(
            validators=[
                InputRequired(),
                EqualTo("confirm", message="Passwords must match"),
            ]
        )
    )
    form_overrides = dict(hashed_password=PasswordField)

    # Permission settings
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True

    # Metadata
    name = "User"
    name_plural = "Users"
    icon = "fa-solid fa-user"
    identity = "user"

    @staticmethod
    def get_category(user: User):
        return "admin" if user.is_admin else "user"

    # --- Permissions ---
    def is_accessible(self, request: Request) -> bool:
        """Check if the current user is authenticated and has access to the admin interface."""
        # Check if the user is authenticated by verifying the JWT token in the session
        token = request.session.get("token")
        if not token:
            logging.warning("No authentication token found in session. Session data: %s", str(request.session))
            return False
        try:
            claims = jwt.decode(token, key=get_settings().secret_key, algorithms=[CryptContextDefaults().ALGORITHM])
            logging.debug("Decoded JWT claims: %s", claims)
            # Optionally, you can also check for specific claims like user category or permissions here
        except ExpiredSignatureError:
            logging.warning("Authentication token has expired. Session data: %s", str(request.session))
            return False
        except JWTError as error:
            logging.warning("Invalid authentication token. Session data: %s. Error: %s", str(request.session), str(error))
            return False
        return True

    def is_visible(self, request: Request) -> bool:
        # Optionally control visibility of the view in the admin sidebar
        return self.is_accessible(request)

    # --- Custom Events ---

    async def on_model_change(
        self, data: dict, model: User, is_created: bool, request: Request
    ) -> None:
        if is_created:
            # Hash the password before saving into DB !
            category = self.get_category(model)
            data["hashed_password"] = pwd_context.hash(
                data["hashed_password"], category=category
            )
        else:
            # If password is being updated, hash it before saving
            if "hashed_password" in data:
                category = self.get_category(model)
                data["hashed_password"] = pwd_context.hash(
                    data["hashed_password"], category=category
                )
        return

    # --- Custom Actions ---
    # ...


class ReportView(BaseView):
    """Custom admin view for displaying reports."""

    name = "Report Page"
    icon = "fa-solid fa-chart-line"

    @expose("/report", methods=["GET"])
    async def report_page(self, request):
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with Session(expire_on_commit=False) as session:  # type: ignore
            stmt = select(func.count(User.id))
            result = await session.execute(stmt)
            users_count = result.scalar_one()

        return await self.templates.TemplateResponse(
            request,
            "report.html",
            context={"users_count": users_count},
        )
