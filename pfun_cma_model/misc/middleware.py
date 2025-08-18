# This file contains the middleware functions for the API.
from fastapi import FastAPI, status, HTTPException, Response
from typing import Optional
import asyncio
from functools import wraps


class UnauthorizedError(HTTPException):
    STATUS_CODE = status.HTTP_401_UNAUTHORIZED


def content_security_policy(csp_policy: Optional[str] = None):
    """
    Decorator to set Content-Security-Policy header on FastAPI endpoints.
    """
    default_csp_policy = "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self';"

    def _collate_csp_policy(additional_csp_policy: Optional[str] = None):
        policies = default_csp_policy.split(";")
        if additional_csp_policy:
            policies += additional_csp_policy.split(";")
        return "; ".join(set([p.strip() for p in policies if p.strip()]))

    def decorator(endpoint):
        @wraps(endpoint)
        async def wrapper(*args, **kwargs):
            response: Response = await endpoint(*args, **kwargs)
            if isinstance(response, Response):
                response.headers["Content-Security-Policy"] = _collate_csp_policy(csp_policy)
            return response
        return wrapper
    return decorator