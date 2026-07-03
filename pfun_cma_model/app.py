"""pfun_cma_model/app.py : pfun-cma-model fastapi app definition."""

from pfun_common.logs import setup_logging
from pfun_common.settings import get_settings

logger = setup_logging()

from pfun_cma_model.api import app
