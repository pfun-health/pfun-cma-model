from sqladmin import ModelView, action
from sqladmin.filters import (
    BooleanFilter,
    AllUniqueStringValuesFilter,
    ForeignKeyFilter,
    OperationColumnFilter,
)
import wtforms
from fastapi import Request
from fastapi.responses import RedirectResponse
from pfun_cma_model.admin.models import *

__all__ = ["UserAdmin"]


# Define User Admin View
class UserAdmin(ModelView, model=User):
    """Admin view for User model."""

    # Save configuration
    save_as = (
        True  # Enable "Save As" functionality to create new records from existing ones
    )

    # Columns configuration
    column_list = ["__all__"]
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

    # Form fields for create/edit views
    form_create_rules = ["name", "hashed_password"]
    form_edit_rules = ["name"]

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

    # --- Custom Events ---

    async def on_model_change(
        self, data: dict, model: User, is_created: bool, request: Request
    ) -> None:
        if is_created:
            # Hash the password before saving into DB !
            data["hashed_password"] = data["hashed_password"] + "_hashed"

    # --- Custom Actions ---

    @action(
        name="approve_users",
        label="Approve",
        confirmation_message="Are you sure?",
        add_in_detail=True,
        add_in_list=True,
    )
    async def approve_users(self, request: Request):
        """Custom action to approve selected users."""
        pks: list[str] = request.query_params.get("pks", "").split(",")
        if pks:
            for pk in pks:
                model: User = await self.get_object_for_edit(pk)  # type: ignore
                ...  # TODO: Implement approval logic, e.g. model.is_approved = True

        referer = request.headers.get("Referer")
        if referer:
            return RedirectResponse(referer)
        else:
            return RedirectResponse(
                request.url_for("admin:list", identity=self.identity)
            )


from pfun_cma_model.app import admin

admin.add_view(UserAdmin)
