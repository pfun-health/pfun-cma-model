from botocore.exceptions import ClientError, ProfileNotFound
import logging
from typing import AnyStr
import boto3

logging.basicConfig(encoding="utf-8", level=logging.WARN)
logger = logging.getLogger()


def set_verbosity(ctx, param, value):
    vdict = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}
    newlevel = vdict.get(value, logger.getEffectiveLevel())
    return newlevel


def get_secret_func(
    secret_name,
    region="us-west-1",
    profile="robbie",
    get_created_date=False,
    verbosity=logging.WARN
):
    #: set logging verbosity
    logger.setLevel(verbosity)

    # Create a Secrets Manager client
    try:
        session = boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:  # handle non-existant profile
        logger.warning(
            "AWS profile %s not found, attempting session without a profile.",
            profile)
        session = boto3.Session(region_name=region)
    finally:
        client = session.client(  # pylint: disable=used-before-assignment
            service_name="secretsmanager",
            region_name=region,
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
            f"...retrieved secret from AWS: '{secret_name}'")

        return secret


def get_secret(
    secret_name,
    region="us-west-1",
    profile="robbie",
    verbosity=logging.WARN,
):
    return get_secret_func(secret_name,
                           region=region,
                           profile=profile,
                           verbosity=verbosity)


def put_secret_func(
    secret_name: AnyStr,
    secret_value: AnyStr,
    region="us-west-1",
    profile="robbie",
    verbosity=logging.WARN,
):
    """set a secret in AWS"""
    #: set logging verbosity
    logger.setLevel(verbosity)

    # Create a Secrets Manager client
    try:
        session = boto3.Session(profile_name=profile, region_name=region)
    except ProfileNotFound:  # handle non-existant profile
        logger.warning(
            f"AWS profile {profile} not found, attempting session without a profile."
        )
        session = boto3.Session()
    finally:
        client = session.client(  # pylint: disable=used-before-assignment
            service_name="secretsmanager",
            region_name=region,
        )

    try:
        client.put_secret_value(SecretId=secret_name,
                                SecretString=secret_value)
    except Exception as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        logging.warning(e, exc_info=True)
        client.create_secret(Name=secret_name, SecretString=secret_value)
    return
