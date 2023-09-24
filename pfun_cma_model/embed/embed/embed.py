import concurrent.futures
import json
import os
import logging
from typing import Optional
import openai
from opensearchpy import OpenSearch, helpers
import certifi
import requests
from sklearn.model_selection import ParameterGrid
import numpy as np
import pfun_path_helper as path_helper

path_helper.append_path(
    os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "../../..")))  # type: ignore
import paramiko
from pfun_cma_model.runtime.chalicelib.engine.cma_model_params import CMAModelParams
from pfun_cma_model.runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from pfun_cma_model.runtime.chalicelib.secrets import get_secret_func as get_secret

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#: OpenSearch Query Type
OpenSearchQuery = str | list[str] | dict


def forward_tunnel(local_port, remote_host, remote_port, ssh_client):
    """
    Creates a forward tunnel from a local port to a remote host and port using an SSH client.

    Parameters:
        local_port (int): The local port to forward.
        remote_host (str): The remote host to forward to.
        remote_port (int): The remote port to forward to.
        ssh_client (SSHClient): The SSH client to use for establishing the tunnel.

    Returns:
        Channel: The channel object representing the established tunnel.
    """
    transport = ssh_client.get_transport()
    channel = transport.open_channel("direct-tcpip",
                                     (remote_host, remote_port),
                                     ("127.0.0.1", local_port))
    return channel


#: set open ai api key
openai.api_key = get_secret("openai-api-key-emacs", region="us-east-1")


class EmbedClient:
    ssh_client: Optional[paramiko.SSHClient] = None  # type: ignore
    tunnel_channel: Optional[paramiko.Channel] = None  # type: ignore
    opensearch_client: Optional[OpenSearch] = None  # type: ignore

    def __del__(self):
        if self.ssh_client is not None:
            self.ssh_client.close()
        if self.tunnel_channel is not None:
            self.tunnel_channel.close()

    def __new__(
        cls,
        ssh_params: Optional[dict] = None,
        opensearch_params: Optional[dict] = None,
        require_ssh_tunnel: bool = False,
    ) -> "EmbedClient":
        if cls.opensearch_client is None:
            cls.opensearch_client = cls.connect_opensearch(
                **opensearch_params or {})
        if cls.ssh_client is None and require_ssh_tunnel:
            cls.setup_ssh_tunnel(**ssh_params or {})
        return super().__new__(cls)

    def __init__(self, *args, **kwds) -> None:
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
        server_config_default = dict(hostname="d2bd",
                                     port=22,
                                     username="robbie")
        server_config = dict(server_config_default, **kwds)

        # tunnel config
        tunnel_config_default = dict(remote_host="10.1.78.132",
                                     local_port=9201,
                                     remote_port=9200)
        tunnel_config = dict(tunnel_config_default, **kwds)

        # Initialize SSH client
        cls.ssh_client = paramiko.SSHClient()
        #: ! add missing host key (only for testing) !
        cls.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Load private key
        private_key_path = kwds.get("private_key_path") or os.path.expanduser(
            "~/.ssh/d2bd_id_rsa")
        privkey = paramiko.RSAKey(filename=private_key_path)
        cls.ssh_client.connect(
            server_config["hostname"],  # type: ignore
            port=server_config["port"],  # type: ignore
            username=server_config["username"],  # type: ignore
            pkey=privkey,
        )

        # Create tunnel
        cls.tunnel_channel = forward_tunnel(
            tunnel_config["local_port"],
            tunnel_config["remote_host"],
            tunnel_config["remote_port"],
            cls.ssh_client,
        )

    @classmethod
    def connect_opensearch(
        cls,
        host: str = "node-0.example.com",
        port: int = 9201,
        username: str = "admin",
        password: str = "admin",
    ):
        auth = (
            username,
            password,
        )  # For testing only. Don't store credentials in code.
        ca_certs_path = str(certifi.where())
        # Create the client with SSL/TLS enabled, but hostname verification disabled.
        cls.opensearch_client = OpenSearch(
            hosts=[{
                "host": host,
                "port": port
            }],
            http_compress=True,  # enables gzip compression for request bodies
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            ca_certs=ca_certs_path,
            timeout=120
        )
        return cls.opensearch_client


class Embedder(EmbedClient):

    def __init__(self, grid_params: Optional[dict] = None, **ssh_params):
        super().__init__(**ssh_params)
        if grid_params is None:
            grid_params = {}
        self.param_grid = self.create_parameter_search_grid(**grid_params)

    @classmethod
    def get_sample_text(cls):
        proc = None
        try:
            api_key = get_secret("pfun-cma-model-aws-api-key",
                                 region="us-east-1")
            sample_text = requests.get(
                "https://api.dev.pfun.app/api/params/default",
                headers={
                    "Content-Type": "application/json",  # type: ignore
                    "Accept": "application/json",
                    "X-Api-Key": api_key,
                },
                timeout=30,
            ).json()
        except Exception as e:
            logger.exception("(%s) Failed to get sample text.",
                             type(e),
                             exc_info=True)
            raise e
        finally:
            if proc is not None:
                proc.terminate()
        return sample_text

    def create_parameter_search_grid(self, num: int = 10, kind='gaussian'):
        param_grid = {}
        cmap = CMAModelParams()
        if hasattr(cmap.bounds, "json"):
            bds = cmap.bounds.json()  # type: ignore
        else:
            bds = cmap.bounds  # type: pfun_cma_model.runtime.chalicelib.engine.bounds.Bounds
        cdict = cmap.model_dump()
        pspace_func = {
            'linear': np.linspace,
            'random': np.random.uniform,
            'gaussian': lambda lb, ub, num: np.random.normal((lb + ub) / 2, (ub - lb) / 2, num),
        }[kind]
        param_grid = {
            k: pspace_func(bds["lb"][j], bds["ub"][j], num)
            for j, k in zip(range(len(cmap.model_fields)),
                            cdict["bounded_param_keys"])
        }  # type: ignore
        param_grid = ParameterGrid(param_grid)
        return param_grid

    @classmethod
    def create_embeddings(cls, text: OpenSearchQuery) -> str:
        model = "text-embedding-ada-002"  # Replace with the model you want to use
        if not isinstance(text, str):
            text = json.dumps(text)
        response = openai.Embedding.create(
            input=text,
            model=model,
        )
        return response.get("data")  # type: ignore

    def save_to_opensearch(self, embedding, doc_id):
        """
        Saves the given embedding and document ID to OpenSearch.

        :param embedding: The embedding to be saved.
        :type embedding: any

        :param doc_id: The ID of the document.
        :type doc_id: str

        :return: The response from the OpenSearch bulk operation.
        :rtype: dict
        """
        action = {
            "_op_type": "index",
            "_index": "embeddings",
            "_id": doc_id,
            "embedding": embedding,
        }
        response = helpers.bulk(self.opensearch_client, [action])
        return response

    def run(self):
        """
        Generates embeddings for each combination of parameters in self.param_grid.
        Prints the progress of creating embeddings for each parameter set.
        Saves the embeddings to OpenSearch and appends the embedding and response to a list.

        Returns:
            embeddings (list): A list of dictionaries containing the embedding and the response from OpenSearch.
        """

        print("Creating embeddings...")

        embeddings = []

        def create_embedding(pset):
            model = CMASleepWakeModel(**pset)
            raw_text = model.run().to_json()
            print(f"\nraw_text:\n\t{raw_text[20:]}...\n")
            embedding = self.create_embeddings(raw_text)
            doc_id = json.dumps(pset)
            _ = self.save_to_opensearch(embedding, doc_id)
            print("...Done creating embedding for " + str(pset))
            return embedding

        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for pset in self.param_grid:
                future = executor.submit(create_embedding, pset)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    embeddings.append(future.result())
                except Exception as e:
                    logging.exception(
                        "(%s) Failed to create embedding.",
                        type(e), exc_info=True
                    )
                else:
                    print('...Done with: %03d / %03d' % (len(embeddings), len(self.param_grid)))
        print()
        print("...done.")
        return embeddings


class EmbedGetter(EmbedClient):
    index_name: str = "embeddings"
    query: Optional[OpenSearchQuery] = None

    def __new__(
        cls,
        ssh_params: Optional[dict] = None,
        opensearch_params: Optional[dict] = None,
        require_ssh_tunnel: bool = True,
        query: Optional[OpenSearchQuery] = None,
        index_name: str = "embeddings",
    ) -> "EmbedGetter":
        """
        Initializes and returns a new instance of the EmbedGetter class.

        Args:
            cls (type): The class object that is being instantiated.
            ssh_params (Optional[dict], optional): The SSH parameters. Defaults to None.
            opensearch_params (Optional[dict], optional): The OpenSearch parameters. Defaults to None.
            query (Optional[OpenSearchQuery], optional): The query parameter. Defaults to None.
            index_name (str, optional): The name of the index. Defaults to 'embeddings'.

        Returns:
            EmbedGetter: A new instance of the EmbedGetter class.
        """
        cls.index_name = index_name
        cls.query = query
        return super().__new__(
            cls,
            ssh_params=ssh_params,
            opensearch_params=opensearch_params,
            require_ssh_tunnel=require_ssh_tunnel,
        )  # type: ignore

    @classmethod
    def retrieve_embeddings(
        cls,
        query: Optional[OpenSearchQuery] = None,
        index_name: str = "embeddings",
        query_type: str = "match_all",
        query_params: Optional[dict] = None,
    ):
        """
        Retrieve embeddings from opensearch.

        Args:
            query (Optional[OpenSearchQuery], optional): The query to retrieve embeddings. Defaults to None.
            index_name (str, optional): The name of the index. Defaults to 'embeddings'.
            query_type (str, optional): The type of query to use. Defaults to 'match_all'.
            query_params (Optional[dict], optional): Additional parameters for search. Defaults to None.

        Returns:
            list: A list of hits containing the retrieved embeddings.
        """
        #: retrieve embeddings from opensearch
        query_params = query_params or {}
        if len(query_params) == 0 and query_type == "fuzzy":
            query_params = {'fuzziness': 'AUTO'}
        if query is None:
            query = cls.query
        if index_name is None:
            index_name = cls.index_name
        if any([query is None, index_name is None]):
            raise RuntimeError("Must provide query and index name!")
        if not isinstance(query, (bytes, str)):
            query = json.dumps(query)
        query_dict = {"embedding": {"value": query, **query_params}}
        body = {"query": {f"{query_type}": {**query_dict}}}
        print('Query:\n', json.dumps(body, indent=3))
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response["hits"]["hits"]


def run_embedder(**kwds):
    """
    Run embedder to create embeddings.

    :return: The embeddings created by the embedder.
    """
    #: run embedder to create embeddings
    embedder = Embedder(**kwds)
    embeddings = embedder.run()  # type: ignore
    print(embeddings[:5])
    return embeddings


def retrieve_embeddings(
    embedder_kwds: Optional[dict] = None, **kwds
):
    """
    Retrieve embeddings from opensearch.

    :param query: A string, or a list of strings, or a dictionary representing the query to retrieve embeddings for. Defaults to the sample text provided by the `Embedder` class.
    :return: A list of embeddings retrieved from opensearch.
    """
    if kwds.get('query') is None:
        kwds['query'] = \
            Embedder.create_embeddings(Embedder.get_sample_text())[0]['embedding']  # type: ignore
    #: retrieve embeddings from opensearch
    embeddings = EmbedGetter(  # pylint: disable=redefined-outer-name
        **embedder_kwds or {}
    ).retrieve_embeddings(**kwds)
    print(embeddings[:5])
    return embeddings


if __name__ == "__main__":
    kwds = dict(query='*', query_type='wildcard')
    embeddings = retrieve_embeddings(
        embedder_kwds=dict(require_ssh_tunnel=False), **kwds)