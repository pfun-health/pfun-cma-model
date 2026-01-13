import logging
import os

__all__ = [
    "settings",
    "Settings",
    "get_settings",
    "utils",
    "setup_logging",
]

try:
    import pfun_common.pfun_common.settings as settings
    import pfun_common.pfun_common.utils as utils
    from pfun_common.pfun_common import setup_logging
except (ImportError, ModuleNotFoundError):
    import pfun_common.settings as settings
    import pfun_common.utils as utils
    from pfun_common.settings import Settings, get_settings
    from pfun_common.utils import setup_logging
