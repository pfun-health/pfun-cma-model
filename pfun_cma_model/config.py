import pfun_path_helper
from pfun_path_helper import get_lib_path
import os
from dataclasses import dataclass

root_path = get_lib_path()


def path_factory(path: str):
    """
    A function that takes a path as input and returns the expanded form of the path.

    Args:
        path (str): The path to be expanded.

    Returns:
        str: The expanded form of the path.
    """
    return os.path.expanduser(path)


@dataclass
class Settings:
    """
    Settings class for the pfun-cma-model package.
    """

    _env_file = os.path.abspath(os.path.join(root_path, "..", ".env"))

    PFUN_APP_SCHEMA_PATH: str = os.getenv(
        "PFUN_APP_SCHEMA_PATH", "~/Git/pfun-app/amplify/backend/api/pfunapp/schema.graphql")
    DEXCOM_CREDS_PATH: str = os.getenv(
        "DEXCOM_CREDS_PATH", "~/.dexcom_creds.json")

    @classmethod
    def load(cls):
        """
        Loads the environment variables from the specified file and creates a new instance of the class.

        :return: An instance of the class with the environment variables loaded.
        :rtype: ClassName
        """
        env = {}
        with open(cls._env_file, "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.strip().split("=")
                env[key] = value
        return env

    def __post_init__(self):
        """
        Initializes the object after it has been created.

        This method updates the object's attributes by loading the data from a file. It iterates through the object's dictionary and checks if the key contains the substring "_PATH" in uppercase. If it does, the value associated with that key is updated with the loaded data.

        Parameters:
            self: The object itself.

        Returns:
            None
        """
        env = self.load()
        for key in env:
            self.__dict__[key] = env[key]
        for key in self.__dict__:
            if "_PATH" in key.upper():
                self.__dict__[key] = path_factory(self.__dict__[key])


settings = Settings()
