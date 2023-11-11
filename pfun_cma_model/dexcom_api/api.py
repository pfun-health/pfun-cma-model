import pfun_path_helper as pph
import os

from pfun_cma_model.dexcom_api.utils import get_creds

lib_path = pph.get_lib_path("dexcom_developer_api_client")
pph.append_path(os.path.abspath(os.path.join(lib_path, "..")))

from dexcom_developer_api_client import Client


class DexcomClient(Client):
    def __init__(self, *args, base_url="https://api.dexcom.com", **kwds) -> None:
        kwds['base_url'] = base_url
        
        super().__init__(*args, **kwds)