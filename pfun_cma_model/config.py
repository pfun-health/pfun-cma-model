import pfun_path_helper
from pfun_path_helper import get_lib_path
import os
from dataclasses import dataclass

root_path = get_lib_path()


@dataclass
class Settings:

    _env_file = os.path.join(root_path, ".env")
    
    PFUN_APP_SCHEMA_PATH: str = os.getenv("PFUN_APP_SCHEMA_PATH")

    @classmethod
    def load(cls):
        env = {}
        with open(cls._env_file, "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split("=")
                env[key] = value
        return cls(**env)


settings = Settings()
