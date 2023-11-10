import json
import os

import pfun_path_helper as pph  # type: ignore
from pfun_cma_model.config import settings


def get_creds():
    """
    Retrieves the Dexcom API credentials from a JSON file and sets the client ID and secret as environment variables.

    Returns:
        Tuple[str, str]: A tuple containing the client ID and client secret.
    """
    creds = json.loads(
        open(settings.DEXCOM_CREDS_PATH, 'r', encoding='utf-8').read()
    )
    os.environ["DEXCOM_CLIENT_ID"] = creds["client_id"]
    os.environ["DEXCOM_CLIENT_SECRET"] = creds["client_secret"]
    return creds["client_id"], creds["client_secret"]
