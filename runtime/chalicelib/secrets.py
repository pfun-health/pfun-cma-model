# Use this code snippet in your app.
# If you need more information about configurations
# or implementing the sample code, visit the AWS docs:
# https://aws.amazon.com/developer/language/python/
from botocore.exceptions import ClientError, ProfileNotFound
from pathlib import Path
import boto3
import logging
from typing import AnyStr
import uuid

logging.basicConfig(encoding="utf-8", level=logging.WARN)
logger = logging.getLogger()


def set_verbosity(ctx, param, value):
    vdict = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}
    newlevel = vdict.get(value, logger.getEffectiveLevel())
    return newlevel


def get_secret_func(
    secret_name,
    region_name="us-west-1",
    profile="robbie",
    get_created_date=False,
    output_fpath=None,
    verbosity=logging.WARN,
):
    #: set logging verbosity
    logger.setLevel(verbosity)

    # Create a Secrets Manager client
    try:
        session = boto3.Session(
            profile_name=profile, region_name=region_name)
    except ProfileNotFound:  # handle non-existant profile
        logger.warning(
            "AWS profile %s not found, attempting session without a profile.",
            profile
        )
        session = boto3.Session()
    finally:
        client = session.client(
            service_name="secretsmanager",
            region_name=region_name,
        )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name)
    except (Exception, ClientError) as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e
    else:
        # Decrypts secret using the associated KMS key.
        secret = get_secret_value_response["SecretString"]

        #: get the secret CreatedDate
        if get_created_date is True:
            secret = (secret, get_secret_value_response["CreatedDate"])

        logger.info(
            f"...retrieved secret from AWS:\n'{secret_name}':\n{secret}\n")

        # write to output path if given
        if output_fpath is not None:
            Path(output_fpath).write_text(secret)
        return secret


def get_secret(
    secret_name,
    region_name="us-west-1",
    profile="robbie",
    output_fpath=None,
    verbosity=logging.WARN,
):
    return get_secret_func(secret_name, region_name, profile, output_fpath, verbosity)


def put_secret_func(
    secret_name: AnyStr,
    secret_value: AnyStr,
    secret_node="secrets.pfun.app",
    region_name="us-west-1",
    profile="robbie",
    verbosity=logging.WARN,
):
    """set a secret in AWS"""
    #: set logging verbosity
    logger.setLevel(verbosity)

    # Create a Secrets Manager client
    try:
        session = boto3.session.Session(
            profile_name=profile, region_name=region_name)
    except ProfileNotFound:  # handle non-existant profile
        logger.warning(
            f"AWS profile {profile} not found, attempting session without a profile."
        )
        session = boto3.session.Session()
    finally:
        client = session.client(
            service_name="secretsmanager",
            region_name=region_name,
        )

    client_request_token = str(
        uuid.uuid3(uuid.uuid4(), secret_node + ":" + secret_name)
    )
    try:
        put_secret_value_response = client.put_secret_value(
            SecretId=secret_name, SecretString=secret_value
        )
    except Exception as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        logging.warn(e, exc_info=1)
        create_secret_response = client.create_secret(
            Name=secret_name, SecretString=secret_value
        )
    return
