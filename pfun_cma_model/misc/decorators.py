from base64 import b64decode, b64encode
from functools import wraps

import numpy as np


def check_is_numpy(f):
    @wraps(f)
    def decorated_function(a, *args, **kwds):
        if not isinstance(a, np.ndarray):
            #: ensure the argument is an array with the input dtype
            a = np.array(a, dtype=type(a[0]))
        return f(a, *args, **kwds)

    return decorated_function
