from typing import (
    Optional,
)
from botocore.config import Config as ConfigCore
import boto3
from botocore.exceptions import ClientError
from threading import Lock


class PFunCMASession:
    """
    A session for interacting with the PFun CMA model resources.
    """

    BOTO3_SESSION = None
    BOTO3_CLIENT = None

    @classmethod
    def get_boto3_session(cls):
        with Lock():
            if cls.BOTO3_SESSION is not None:
                return cls.BOTO3_SESSION
            cls.BOTO3_SESSION = boto3.Session()
        return cls.BOTO3_SESSION

    @classmethod
    def get_boto3_client(cls, service_name: str, session: Optional[boto3.Session] = None, *args, **kwds):
        """
        Creates a new Boto3 client for a specified AWS service.

        Args:
            service_name (str): The name of the AWS service for which the client is being created.
            session (boto3.Session, optional): An existing Boto3 session to use. If not provided, a new session will be created.
            *args: Additional arguments that will be passed to the Boto3 client constructor.
            **kwds: Additional keyword arguments that will be passed to the Boto3 client constructor.

        Returns:
            boto3.client: The newly created Boto3 client for the specified AWS service.
        """
        with Lock():
            config = ConfigCore(region_name='us-east-1')
            if session is None:
                session = cls.get_boto3_session()
            client = session.client(service_name, *args, config=config, **kwds)
        return client