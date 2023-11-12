import json
import os

import pfun_path_helper as pph  # type: ignore
from pfun_cma_model.config import settings
from pfun_cma_model.pfun_secrets import get_secret


def get_creds(from_aws=True, app_name="pfun-fiona"):
    """
    Retrieves the Dexcom API credentials from a JSON file and sets the client ID and secret as environment variables.

    Returns:
        Tuple[str, str]: A tuple containing the client ID and client secret.
    """
    if from_aws is True:
        value = get_secret("pfun-dexcom-creds")
        creds = json.loads(value)
    else:
        creds = json.loads(
            open(settings.DEXCOM_CREDS_PATH, 'r', encoding='utf-8').read()
        )
    creds = creds.get(app_name)
    if creds is None:
        raise ValueError(f"Could not find credentials for app '{app_name}'")
    os.environ["DEXCOM_CLIENT_ID"] = creds["client_id"]
    os.environ["DEXCOM_CLIENT_SECRET"] = creds["client_secret"]
    return creds["client_id"], creds["client_secret"]
