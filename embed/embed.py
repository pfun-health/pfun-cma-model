import os
import openai
from opensearchpy import OpenSearch, helpers
import certifi
import uuid
import subprocess
import threading
import requests
from sklearn.model_selection import ParameterGrid
import numpy as np
import pfun_path_helper as path_helper
path_helper.append_path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from runtime.chalicelib.engine.cma_model_params import CMAModelParams
from runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from runtime.chalicelib.secrets import get_secret_func as get_secret
import paramiko
import select

def forward_tunnel(local_port, remote_host, remote_port, ssh_client):
    transport = ssh_client.get_transport()
    channel = transport.open_channel(
        "direct-tcpip",
        (remote_host, remote_port),
        ("127.0.0.1", local_port)
    )
    return channel


#: set open ai api key
openai.api_key = get_secret('openai-api-key-emacs', region='us-east-1')


class Embedder:
    def __init__(self, ssh_params={}):
        self.ssh_client = None
        self.tunnel_channel = None
        # self.setup_ssh_tunnel(**ssh_params)
        self.completed_proc = None
        self.opensearch_client = self.connect_opensearch()
        self.param_grid = self.create_parameter_search_grid()

    def setup_ssh_tunnel(self, **kwds):
        # ssh server config
        server_config = dict(
            hostname = 'd2bd',
            port = 22,
            username = 'robbie'
        )
        # tunnel config
        tunnel_config = dict(
            remote_host = '10.1.78.150',
            local_port = 9201,
            remote_port = 9201
        )
        # Initialize SSH client
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # Load private key
        private_key_path = os.path.expanduser('~/.ssh/d2bd_id_rsa')
        privkey = paramiko.RSAKey(filename=private_key_path)
        self.ssh_client.connect(
            server_config['hostname'], port=server_config['port'], username=server_config['username'], pkey=privkey)
        # Create tunnel
        self.tunnel_channel = forward_tunnel(tunnel_config['local_port'], tunnel_config['remote_host'], tunnel_config['remote_port'], self.ssh_client)

    def get_sample_text(self):
        sample_text = requests.get('api.dev.pfun.app/api/params/default', headers={'Content-Type': 'application/json'}, timeout=5).json()
        return sample_text

    def create_parameter_search_grid(self):
        param_grid = {}
        cmap = CMAModelParams()
        if hasattr(cmap.bounds, 'json'):
            bds = cmap.bounds.json()  # type: ignore
        else:
            bds = cmap.bounds  # type: ignore
        cdict = cmap.model_dump()
        param_grid = {k: np.linspace(bds['lb'][j], bds['ub'][j], num=5) for j, k in zip(range(len(cmap.model_fields)), cdict.get('bounded_param_keys'))}  # type: ignore
        param_grid = ParameterGrid(param_grid)
        return param_grid

    def connect_opensearch(self):
        auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
        ca_certs_path = certifi.where()

        # Create the client with SSL/TLS enabled, but hostname verification disabled.
        self.opensearch_client = OpenSearch(
            hosts=[{'host': 'node-0.example.com', 'port': 9201}],
            http_compress=True,  # enables gzip compression for request bodies
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            ca_certs=ca_certs_path
        )
        return self.opensearch_client

    def get_embeddings(self, text) -> str:
        model = "text-embedding-ada-002"  # Replace with the model you want to use
        response = openai.Embedding.create(input=text, model=model, )
        return response.get('data')

    def save_to_opensearch(self, embedding, doc_id):
        action = {
            "_op_type": "index",
            "_index": "embeddings",
            "_id": doc_id,
            "embedding": embedding
        }
        response = helpers.bulk(self.opensearch_client, [action])
        return response

    def run(self):
        embeddings = []
        print('Creating embeddings...')
        for pset in self.param_grid:
            print('\nCreating embedding for:\n\t' + str(pset))
            raw_text = CMASleepWakeModel(**pset).json()
            print(f'\nraw_text:\n\t{raw_text}\n')
            embedding = self.get_embeddings(raw_text)
            doc_id = "embedding-" + str(uuid.uuid4())
            response = self.save_to_opensearch(embedding, doc_id)
            embeddings.append({'embedding': embedding, 'response': response})
            print('...Done creating embedding for ' + str(pset))
        print()
        print('...done.')
        return embeddings


if __name__ == "__main__":
    embedder = Embedder()
    embeddings = embedder.run()
    embedder.ssh_client.close()
    print(embeddings[:5])