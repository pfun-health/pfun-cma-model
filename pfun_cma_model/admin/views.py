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
from pfun_cma_model.admin.models import *
from pfun_cma_model.admin.core import Base, engine, Session

__all__ = ["UserAdmin", "ReportView"]

"""pfun_cma_model/admin/views.py : Admin views for pfun-cma-model."""


# Define User Admin View
class UserAdmin(ModelView, model=User):
    """Admin view for User model."""

    # Save configuration
    save_as = (
        True  # Enable "Save As" functionality to create new records from existing ones
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

    # --- Permissions ---
    def is_accessible(self, request: Request) -> bool:
        # Implement your authentication logic here
        # For example, check if the user is logged in and has admin privileges
        # TODO: Implement real authentication logic
        token = request.session.get("token")
        if not token:
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
            data["hashed_password"] = data["hashed_password"] + "_hashed"

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
