"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""

from pfun_common.settings import get_settings
from pfun_cma_model.api import app


###
# --- Setup Admin (sqladmin) ---
###

from pfun_cma_model.admin.core import engine
from sqladmin import Admin

# Configure the admin interface with the SQLAlchemy engine and register views
admin = Admin(app, engine)
