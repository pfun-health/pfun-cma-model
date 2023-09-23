"""
PFun CMA Model Frontend.
"""
import os
import json
import sys
import uuid
from chalice.app import (
    ConvertToMiddleware,
    Request, BadRequestError, Chalice,
    Response, CORSConfig
)
from requests import post
from requests.sessions import Session
from pathlib import Path
import urllib.parse as urlparse
from typing import (
    Any, Dict, Literal, Optional
)
from botocore.exceptions import ClientError
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pfun_path_helper
pfun_path_helper.append_path(Path(__file__).parent.parent)
import runtime.chalicelib.utils as utils
from runtime.chalicelib.pathdefs import (
    FRONTEND_ROUTES,
    PUBLIC_ROUTES,
    PRIVATE_ROUTES
)
from runtime.chalicelib.secrets import get_secret_func
from runtime.chalicelib.sessions import PFunCMASession
from runtime.chalicelib.middleware import authorization_required as authreq
from functools import partial
