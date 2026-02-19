"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""

from pfun_common.settings import get_settings
from pfun_cma_model.api import app


###
# --- Setup Admin (sqladmin) ---
###

from pfun_cma_model.admin.core import engine
from pfun_cma_model.admin.views import *
from sqladmin import Admin

admin = Admin(app, engine)

admin.add_view(UserAdmin)
