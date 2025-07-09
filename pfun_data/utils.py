"""pfun_data.utils
"""
import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml


def get_data_dirpath() -> Path:
    """Get the path to the data directory.

    Returns:
        Path: The path to the data directory.
    """
    return Path(__file__).parent