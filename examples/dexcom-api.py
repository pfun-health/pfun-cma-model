# %%
# %pip install flask requests flask_oauthlib

# %%
import os
import json
creds = json.loads(open(os.path.expanduser('~/.credentials/dexcom_pfun-app_glucose.json')).read())
os.environ['DEXCOM_CLIENT_ID'] = creds['client_id']
os.environ['DEXCOM_CLIENT_SECRET'] = creds['client_secret']

# %%
import os
from flask import Flask, redirect, url_for, session, request
from flask_oauthlib.client import OAuth

app = Flask(__name__)
app.secret_key = 'random_secret_key'  # Change this to a random secret key
oauth = OAuth(app)

dexcom = oauth.remote_app(
    'dexcom',
    consumer_key=os.getenv('DEXCOM_CLIENT_ID'),
    consumer_secret=os.getenv('DEXCOM_CLIENT_SECRET'),
    base_url='https://api.dexcom.com/v2/',
    request_token_url='https://api.dexcom.com/v2/oauth2/token',
    access_token_method='POST',
    access_token_url='https://api.dexcom.com/v2/oauth2/token',
    authorize_url='https://api.dexcom.com/v2/oauth2/login'
)


@app.route('/')
def index():
    return 'Welcome to Dexcom Wrapper! <a href="/login">Login with Dexcom</a>'

@app.route('/login')
def login():
    return dexcom.authorize(callback=url_for('authorized', _external=True))

@app.route('/login/authorized')
def authorized():
    response = dexcom.authorized_response()
    if response is None or response.get('access_token') is None:
        return 'Could not authorize', 403

    session['dexcom_token'] = (response['access_token'], '')
    me = dexcom.get('users/self')
    return 'Logged in as: {}'.format(me.data)

@dexcom.tokengetter
def get_dexcom_oauth_token():
    return session.get('dexcom_token')

if __name__ == '__main__':
    app.run(debug=True, port=5000)


# %%



