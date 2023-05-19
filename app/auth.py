import boto3
from threading import Thread


class IAMAuthSession(Thread):
    def __init__(self, *args, **kwds):
        Thread.__init__(*args, **kwds)
        #: ref: initiate-auth... https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/cognito-idp/client/initiate_auth.html
        #: ref: How to authenticate... https://stackoverflow.com/a/63388491/1871569
        #: tutorial: https://medium.com/@houzier.saurav/aws-cognito-with-python-6a2867dd02c6
        self.client = boto3.client('cognito-idp')

    def newUser(self, username, ...):
        #: signup: https://medium.com/@houzier.saurav/aws-cognito-with-python-6a2867dd02c6
        user_resp = self.client.create_user(UserName=username)
        key_resp = self.client.create_access_key(UserName=username)
        #: TODO ... store in secure database, e.g. boto3.secrets
        return user_resp, key_resp

    def attemptLogin(self, session, ...):
        pass

    def run(self, access_key, secret_key, region, profile=None):
        #: create new session
        session = boto3.session.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region, profile=profile) 

        #: ...resources... `s3 = session.resource('s3')`, e.g.

        # use threadsafe resources,
        #: e.g., resource = session.resource('cognito-idp'); resource.admin_initiate_auth(...)

        pass
