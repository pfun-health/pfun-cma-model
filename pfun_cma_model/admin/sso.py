"""pfun_cma_model/admin/sso.py : SSO authentication backend for sqladmin."""
import logging
logger = logging.getLogger(__name__)
from fastapi_sso.sso.base import OpenID
from fastapi_sso.sso.google import GoogleSSO
from packages.pfun_common.pfun_common.settings import get_settings
import urllib.parse as urlparse
logger.setLevel(level=logging.DEBUG if get_settings().debug is True else logging.INFO)

def setup_google_sso(
    redirect_host: str, redirect_path: str = "/sso/auth/callback"
) -> GoogleSSO:
    """Setup the Google SSO authentication backend."""
    logger.debug(
        "Setting up Google SSO with:\n\t\t\t+ redirect host: %s\n\t\t\t+ redirect path: %s",
        redirect_host, redirect_path
    )
    #: Ensure the redirect URI is properly formed (scheme is required for urljoin to work correctly)
    redirect_base_url = str(redirect_host)
    if not redirect_base_url.startswith(("http://", "https://")):
        redirect_base_url = "https://" + redirect_base_url
    redirect_uri = str(urlparse.urljoin(redirect_base_url, redirect_path))
    logger.debug("Setting up Google SSO with redirect URI: %s", redirect_uri)
    return GoogleSSO(
        client_id=get_settings().google_cloud_client_id,
        client_secret=get_settings().google_cloud_client_secret,
        redirect_uri=redirect_uri,
    )
