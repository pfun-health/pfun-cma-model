# This file contains the middleware functions for the API.
from fastapi import FastAPI, Request, status, HTTPException, Response
from functools import wraps


class UnauthorizedError(HTTPException):
    STATUS_CODE = status.HTTP_401_UNAUTHORIZED


def authorization_required(
    app: FastAPI,
    get_current_request,
    PRIVATE_ROUTES,
    SECRETS_CLIENT,
    PFunCMASession,
    logger,
):
    """
    A wrapper function that handles authentication for the API.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of the wrapped function.

    Raises:
        UnauthorizedError: If the authentication parameters are invalid.

    """

    def wrapper_outer(
        func,
        app=app,
        get_current_request=get_current_request,
        PRIVATE_ROUTES=PRIVATE_ROUTES,
        SECRETS_CLIENT=SECRETS_CLIENT,
        PFunCMASession=PFunCMASession,
        logger=logger,
    ):

        @wraps(func)
        def wrapper(*args,
                    app=app,
                    get_current_request=get_current_request,
                    PRIVATE_ROUTES=PRIVATE_ROUTES,
                    SECRETS_CLIENT=SECRETS_CLIENT,
                    PFunCMASession=PFunCMASession,
                    logger=logger,
                    **kwargs):
            if not hasattr(app, "current_request"):
                #: skip authorization for lambda functions
                return func(*args, **kwargs)
            current_request = get_current_request()
            if current_request.path not in PRIVATE_ROUTES:
                #: skip authorization for public routes
                return func(*args, **kwargs)
            if SECRETS_CLIENT is None:  # type: ignore  # noqa: F823
                # lazy load secrets client
                SECRETS_CLIENT = PFunCMASession.get_boto3_client(
                    "secretsmanager")
            api_key = SECRETS_CLIENT.get_secret_value(  # type: ignore
                SecretId="pfun-cma-model-aws-api-key")["SecretString"]
            rapidapi_key = SECRETS_CLIENT.get_secret_value(  # type: ignore
                SecretId="pfun-cma-model-rapidapi-key")["SecretString"]
            logger.info("RapidAPI key: %s", rapidapi_key)
            logger.info("API key: %s", api_key)
            api_key_given = current_request.headers.get("X-API-Key")
            apikey_authorized = api_key_given == api_key
            rapidapi_key_given = current_request.headers.get("X-RapidAPI-Key")
            rapidapi_authorized = rapidapi_key_given == rapidapi_key
            try:
                if any([apikey_authorized, rapidapi_authorized]):
                    logger.info("Authorized request: %s",
                                str(vars(current_request)))
                    return func(*args, **kwargs)
                else:
                    raise UnauthorizedError("Unauthorized request: %s" %
                                            str(vars(current_request)))
            except UnauthorizedError:
                logger.error(
                    "authorization parameters given:\n\tapi_key: %s (%s),\n\trapidapi_key: %s (%s)",
                    api_key_given,
                    str(apikey_authorized),
                    rapidapi_key_given,
                    str(rapidapi_authorized),
                )
                return Response(
                    content=
                    "Unauthorized request.\nAuth params:\n\tapi_key: %s,\n\trapidapi_key: %s"
                    % (api_key_given, rapidapi_key_given),
                    status_code=401,
                )

        return wrapper

    return wrapper_outer
