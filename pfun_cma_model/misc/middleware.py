# This file contains the middleware functions for the API.
import asyncio
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Optional

from fastapi import HTTPException, Request, Response, status


async def track_client_request_middleware(request: Request, call_next):
    """
    Middleware to track client requests and store them in Redis.

    Stores: {client_ip: {cookies, session_id, request_info, timestamp}}
    """
    from pfun_cma_model.app import redis_client  # Avoid circular import

    # Extract client IP address
    client_ip = request.client.host if request.client else "unknown"

    # Extract cookies
    cookies = dict(request.cookies) if request.cookies else {}

    # Extract session ID if available
    session_id = (
        request.session.get("session_id") if hasattr(request, "session") else None
    )

    # Extract request details
    query_params = dict(request.query_params) if request.query_params else {}

    # Build request info object
    request_info = {
        "client_ip": client_ip,
        "remote_ip": request.headers.get(
            "X-Forwarded-For", request.headers.get("X-Real-IP", client_ip)),
        "cookies": cookies,
        "session_id": session_id,
        "method": request.method,
        "path": request.url.path,
        "query_params": query_params,
        "timestamp": datetime.now().isoformat(),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_agent": request.headers.get("user-agent", "unknown"),
        "referer": request.headers.get("referer", None),
    }

    # Store in Redis if available
    if redis_client is None:
        logging.debug(
            "Redis not connected. Client request info (debug only): IP=%s, Request=%s",
            client_ip,
            request_info,
        )
        return await call_next(request)
    elif redis_client is not None:
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            redis_key = (
                f"client_request:{client_ip}:{session_id or 'no-session'}:{timestamp}"
            )
            request_json = json.dumps(request_info, default=str)
            await redis_client.set(
                redis_key, request_json, ex=3600  # Expire after 1 hour
            )
            logging.debug(
                "Client request tracked in Redis: IP=%s, Method=%s, Path=%s",
                client_ip,
                request.method,
                request.url.path,
            )
        except Exception as exc:
            logging.warning(
                "Failed to store client request in Redis: %s", str(exc), exc_info=True
            )

    # Continue with the request
    response = await call_next(request)
    return response


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
                response.headers["Content-Security-Policy"] = _collate_csp_policy(
                    csp_policy
                )
            return response

        return wrapper

    return decorator
