import os

__all__ = [
    "settings",
    "Settings",
    "get_settings",
    "utils",
    "load_environment_variables",
    "setup_logging"
]

try:
    import pfun_common.pfun_common.settings as settings
    import pfun_common.pfun_common.utils as utils
    from pfun_common.pfun_common import (
        load_environment_variables,
        setup_logging
    )
except (ImportError, ModuleNotFoundError):
    import pfun_common.settings as settings
    from pfun_common.settings import (
        Settings, get_settings
    )
    import pfun_common.utils as utils
    from pfun_common.utils import (
        load_environment_variables,
        setup_logging
    )