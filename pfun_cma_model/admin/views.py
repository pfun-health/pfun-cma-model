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

    # Columns to display in list view
    column_list = ["id", "name", "email", "is_admin", "age"]
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

    # Form settings
    form_columns = [User.name]
    form_args = dict(name=dict(label="Full name"))
    form_widget_args = dict(email=dict(readonly=True))
    form_overrides = dict(email=wtforms.EmailField)
    form_include_pk = True
    form_ajax_refs = {
        "address": {
            "fields": ("zip_code", "street"),
            "order_by": ("id",),
        }
    }
    form_create_rules = ["name", "password"]
    form_edit_rules = ["name"]

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
