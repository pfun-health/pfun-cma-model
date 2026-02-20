"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""

from pfun_common.settings import get_settings
from pfun_cma_model.api import app

###
# --- Setup Admin (sqladmin) ---
###
from pfun_cma_model.admin.core import engine
from pfun_cma_model.admin.auth import authentication_backend
from sqladmin import Admin

# Configure the admin interface with the SQLAlchemy engine and register views
admin = Admin(
    app, engine, authentication_backend=authentication_backend, title="PFun CMA Admin"
)

# Import admin views to register them with the admin interface
from pfun_cma_model.admin.views import UserAdmin, ReportView

admin.add_view(UserAdmin)
admin.add_view(ReportView)
