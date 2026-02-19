from fastapi import FastAPI
import logging


def mount_adminsite_instance(app: FastAPI) -> FastAPI:
    """Create, configure, and mount an AdminSite instance.
    """
    try:
        # Import lazily to avoid hard dependency import-time failures
        from fastapi_user_auth.admin.site import AuthAdminSite

        # Import Admin settings lazily (may trigger imports of admin deps)
        from fastapi_amis_admin.admin import Settings as AdminSettings

        # Create AdminSite instance
        site = AuthAdminSite(
            settings=AdminSettings(database_url_async="sqlite+aiosqlite:///amisadmin.db")
        )
        # Mount the AdminSite instance to the FastAPI app
        site.mount_app(app)
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Admin site not mounted: %s",
            exc,
        )
    return app