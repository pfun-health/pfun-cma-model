import os
from typing import Optional
import openai
from opensearchpy import OpenSearch, helpers
import certifi
import uuid
import requests
from sklearn.model_selection import ParameterGrid
import numpy as np
import pfun_path_helper as path_helper
path_helper.append_path(os.path.abspath(
    os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..')))  # type: ignore
from runtime.chalicelib.engine.cma_model_params import CMAModelParams
from runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from runtime.chalicelib.secrets import get_secret_func as get_secret
import paramiko


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


class EmbedClient:

    ssh_client: Optional[paramiko.SSHClient] = None  # type: ignore
    tunnel_channel: Optional[paramiko.Channel] = None  # type: ignore
    opensearch_client: Optional[OpenSearch] = None  # type: ignore

    def __del__(self):
        if self.ssh_client is not None:
            self.ssh_client.close()
        if self.tunnel_channel is not None:
            self.tunnel_channel.close()

    def __new__(cls,
                ssh_params: Optional[dict] = None,
                opensearch_params: Optional[dict] = None,
                require_ssh_tunnel: bool = True
                ) -> "EmbedClient":
        if cls.opensearch_client is None:
            cls.opensearch_client = cls.connect_opensearch(**opensearch_params or {})
        if cls.ssh_client is None and require_ssh_tunnel:
            cls.setup_ssh_tunnel(**ssh_params or {})
        return super().__new__(cls)

    def __init__(self, ) -> None:
        self.completed_proc = None
        self.opensearch_client: OpenSearch = self.opensearch_client
        self.ssh_client: paramiko.SSHClient = self.ssh_client
        self.tunnel_channel: paramiko.Channel = self.tunnel_channel

    @classmethod
    def setup_ssh_tunnel(cls, **kwds):
        """Setup SSH tunnel to remote server.

        Keyword Arguments:
        ------------------
            #: SSH Server Config:
            private_key_path (str): path to private key
            hostname (str): hostname of remote server
            port (int): port of remote server
            username (str): username for remote server

            #: Tunnel Config:
            remote_host (str): hostname of remote host
            remote_port (int): port of remote host
            local_port (int): local port to forward to remote host
        """

        # ssh server config
        server_config_default = dict(
            hostname='d2bd',
            port=22,
            username='robbie'
        )
        server_config = dict(server_config_default, **kwds)

        # tunnel config
        tunnel_config_default = dict(
            remote_host='10.1.78.132',
            local_port=9201,
            remote_port=9200
        )
        tunnel_config = dict(tunnel_config_default, **kwds)

        # Initialize SSH client
        cls.ssh_client = paramiko.SSHClient()
        #: ! add missing host key (only for testing) !
        cls.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load private key
        private_key_path = kwds.get('private_key_path') or os.path.expanduser('~/.ssh/d2bd_id_rsa')
        privkey = paramiko.RSAKey(filename=private_key_path)
        cls.ssh_client.connect(
            server_config['hostname'],  # type: ignore
            port=server_config['port'],  # type: ignore
            username=server_config['username'],  # type: ignore
            pkey=privkey)

        # Create tunnel
        cls.tunnel_channel = forward_tunnel(
            tunnel_config['local_port'],
            tunnel_config['remote_host'],
            tunnel_config['remote_port'],
            cls.ssh_client
        )

    @classmethod
    def connect_opensearch(cls,
                           host: str = 'node-0.example.com',
                           port: int = 9201,
                           username: str = 'admin',
                           password: str = 'admin'
                           ):
        auth = (username, password)  # For testing only. Don't store credentials in code.
        ca_certs_path = str(certifi.where())
        # Create the client with SSL/TLS enabled, but hostname verification disabled.
        cls.opensearch_client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,  # enables gzip compression for request bodies
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            ca_certs=ca_certs_path
        )
        return cls.opensearch_client


class Embedder(EmbedClient):
    def __init__(self, **ssh_params):
        super().__init__(**ssh_params)
        self.param_grid = self.create_parameter_search_grid()

    @classmethod
    def get_sample_text(cls):
        api_key = get_secret('pfun-cma-model-aws-api-key', region='us-east-1')
        sample_text = requests.get(
            'https://api.dev.pfun.app/api/params/default',
            headers={'Content-Type': 'application/json',  # type: ignore
                     'Accept': 'application/json',
                     'X-Api-Key': api_key}, timeout=5
        ).json()
        return sample_text

    def create_parameter_search_grid(self, num: int = 5):
        param_grid = {}
        cmap = CMAModelParams()
        if hasattr(cmap.bounds, 'json'):
            bds = cmap.bounds.json()  # type: ignore
        else:
            bds = cmap.bounds  # type: ignore
        cdict = cmap.model_dump()
        param_grid = {k: np.linspace(bds['lb'][j], bds['ub'][j], num=num) for j, k in zip(range(len(cmap.model_fields)), cdict.get('bounded_param_keys'))}  # type: ignore
        param_grid = ParameterGrid(param_grid)
        return param_grid

    @classmethod
    def create_embeddings(cls, text: str | list[str] | dict) -> str:
        model = "text-embedding-ada-002"  # Replace with the model you want to use
        response = openai.Embedding.create(input=text, model=model, )
        return response.get('data')  # type: ignore

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
            model = CMASleepWakeModel(**pset)
            raw_text = model.run().to_json()
            print(f'\nraw_text:\n\t{raw_text[20:]}...\n')
            embedding = self.create_embeddings(raw_text)
            doc_id = "embedding-" + str(uuid.uuid4())
            response = self.save_to_opensearch(embedding, doc_id)
            embeddings.append({'embedding': embedding, 'response': response})
            print('...Done creating embedding for ' + str(pset))
        print()
        print('...done.')
        return embeddings


class EmbedGetter(EmbedClient):

    index_name: str = 'embeddings'
    query: Optional[str | list[str] | dict] = None

    def __new__(cls,
                ssh_params: Optional[dict] = None,
                opensearch_params: Optional[dict] = None,
                query: Optional[str | list[str] | dict] = None,
                index_name: str = 'embeddings'
                ) -> "EmbedGetter":
        cls.index_name = index_name
        cls.query = query
        return super().__new__(cls)  # type: ignore

    @classmethod
    def retrieve_embeddings(cls,
                            query: Optional[str | list[str] | dict] = None,
                            index_name: str = 'embeddings',
                            fuzzy: bool = False,
                            fuzzy_params: Optional[dict] = None):
        #: retrieve embeddings from opensearch
        fuzzy_params = fuzzy_params if fuzzy_params is not None else \
            {"fuzziness": "AUTO" if fuzzy else 0}
        if query is None:
            query = cls.query
        if index_name is None:
            index_name = cls.index_name
        if any([query is None, index_name is None]):
            raise RuntimeError("Must provide query and index name!")
        query_dict = {
            "field_name": {
                "query": query,
                **fuzzy_params
            }
        }
        body = {
            "query": {
                "match": {
                    **query_dict
                }
            }
        }
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response['hits']['hits']


def run_embedder():
    #: run embedder to create embeddings
    embedder = Embedder()
    embeddings = embedder.run()  # type: ignore
    print(embeddings[:5])
    return embeddings


def retrieve_embeddings(query: str | list[str] | dict = Embedder.get_sample_text()):
    #: retrieve embeddings from opensearch
    embeddings = EmbedGetter().retrieve_embeddings(query)
    print(embeddings[:5])
    return embeddings
