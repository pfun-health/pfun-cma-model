import concurrent.futures
import json
import subprocess
from typing import Sequence
import click
from typing import Optional, List
import os
import logging
from pandas import DataFrame, Series
import tiktoken
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
from pfun_cma_model.runtime.src.engine.cma_model_params import CMAModelParams
from pfun_cma_model.runtime.src.engine.cma_sleepwake import CMASleepWakeModel
from pfun_cma_model.secrets import get_secret_func as get_secret
from pfun_cma_model.runtime.src.engine.cma_model_params import Bounds
from pfun_cma_model.runtime.src.engine.data_utils import (
    downsample_data,
    interp_missing_data,
)
from pfun_cma_model.runtime.src.engine.calc import normalize

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

#: OpenSearch Query Type
OpenSearchQuery = str | list[str] | dict

#: get tiktoken encoding for text-embedding-ada-002
encoding = tiktoken.encoding_for_model("text-embedding-ada-002")


def get_opensearch_host():
    return subprocess.check_output(
        "kubectl get endpoints/opensearch-deployment-master-hl --output json | jq '.subsets[0].addresses[0].ip'", shell=True).decode("utf-8").strip().replace('"', '')


def encode(text: str | Sequence[str],
           do_norm=True,
           threads: int = 4) -> Sequence[float | int | Sequence[float | int]]:
    if isinstance(text, str):
        text = [text]
    text = [json.dumps(t) if not isinstance(t, str) else t for t in text]
    out = encoding.encode_batch(text, num_threads=threads)  # type: ignore
    if do_norm is False:
        return out
    return [normalize(o).tolist() for o in out]


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
        **kwds,
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
        remote_host = kwds.get('remote_host')
        if remote_host is None:
            remote_host = get_opensearch_host()
        tunnel_config_default = dict(remote_host=remote_host,
                                     local_port=9200,
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
        host: Optional[str] = None,
        port: int = 9200,
        username: str = "admin",
        password: Optional[str] = None,
    ):
        if password is None:
            password = subprocess.check_output('pass show opensearch/admin', shell=True).decode("utf-8").strip()
        if host is None:
            host = get_opensearch_host()
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
            timeout=120,
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

    def create_parameter_search_grid(self, num: int = 10, kind="gaussian"):
        param_grid = {}
        cmap = CMAModelParams()
        if hasattr(cmap.bounds, "json"):
            bds = cmap.bounds.json()  # type: ignore
        else:
            bds = cmap.bounds
        cdict = cmap.model_dump()
        pspace_func = {
            "linear":
            np.linspace,
            "random":
            np.random.uniform,
            "gaussian":
            lambda lb, ub, num: np.random.normal((lb + ub) / 2,
                                                 (ub - lb) / 2, num),
        }[kind]
        param_grid = {
            k: pspace_func(bds["lb"][j], bds["ub"][j], num)  # type: ignore
            for j, k in zip(range(len(cmap.model_fields)),
                            cdict["bounded_param_keys"])
        }  # type: ignore
        param_grid = ParameterGrid(param_grid)
        return param_grid

    @classmethod
    def create_tokenized_embeddings(cls, text: str | Sequence[str], **kwds):
        return encode(text, **kwds)

    @classmethod
    def create_timeseries_embeddings(cls, df: DataFrame | Series, **kwds):
        df = downsample_data(df)
        df = interp_missing_data(df)
        #: convert embedding to byte integers
        embedding = df.to_numpy(dtype=float, na_value=0.0).flatten()
        embedding = normalize(100.0 * embedding, a=-127.0, b=127.0)
        embedding = embedding.astype("int8")
        return embedding.tolist()

    def save_to_opensearch(self, doc_id, embedding):
        """
        Saves the given embedding and document ID to OpenSearch.

        :param doc_id: The ID of the document.
        :type doc_id: str
        :param embedding: The embedding to be saved.
        :type embedding: any

        :return: The response from the OpenSearch bulk operation.
        :rtype: dict
        """

        def format2action(params, embedding):
            doc_id = params
            if not isinstance(doc_id, str):
                doc_id = json.dumps(doc_id)
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            if isinstance(params, str):
                params = json.loads(params)
            action = {
                "_index": "embeddings",
                "_id": doc_id,
                "_source": {
                    "embedding": embedding,
                    "cma_params": params
                },
            }
            return action

        actions = []
        if isinstance(embedding, list) and isinstance(doc_id, list):
            for params, em in zip(doc_id, embedding):
                action = format2action(params, em)
                actions.append(action)
        else:
            actions.append(format2action(doc_id, embedding))
        response = helpers.bulk(self.opensearch_client, actions)
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

        doc_ids = []
        embeddings = []

        def create_embedding(pset):
            model = CMASleepWakeModel(**pset)
            soln = model.run()["G"]
            #: end up with example 1024 samples
            embedding = self.create_timeseries_embeddings(soln)
            doc_id = json.dumps(pset)
            return (doc_id, embedding)

        futures = []
        batch_size = min(1000, len(self.param_grid))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for pset in self.param_grid:
                future = executor.submit(create_embedding, pset)
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                try:
                    doc_id, embedding = future.result()
                except Exception as e:
                    logging.exception("(%s) Failed to create embedding.",
                                      type(e),
                                      exc_info=True)
                else:
                    if len(embeddings) < batch_size - 1:
                        doc_ids.append(doc_id)
                        embeddings.append(embedding)
                    else:
                        print("Saving embeddings to OpenSearch...")
                        self.save_to_opensearch(doc_ids, embeddings)
                        print("...Done with: %03d / %03d" %
                              (len(embeddings), len(self.param_grid)))
                        embeddings = []  # reset embeddings
                        print("...done with batch.")
                        print()
        print()
        print("...done.")
        return {"doc_ids": doc_ids, "embeddings": embeddings}


class EmbedGetter(EmbedClient):
    index_name: str = "embeddings"
    query: Optional[OpenSearchQuery] = None

    def __new__(
        cls,
        ssh_params: Optional[dict] = None,
        opensearch_params: Optional[dict] = None,
        require_ssh_tunnel: bool = False,
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
        obj = super().__new__(
            cls,
            ssh_params=ssh_params,
            opensearch_params=opensearch_params,
            require_ssh_tunnel=require_ssh_tunnel,
        )  # type: ignore
        obj.query = query
        return obj

    @classmethod
    def retrieve_embeddings(
        cls,
        query: Optional[OpenSearchQuery] = None,
        index_name: str = "embeddings",
        query_type: str = "match_all",
        query_size: int = 1,
        query_params: Optional[dict] = None,
    ) -> List[dict]:
        """
        Retrieve embeddings from opensearch.

        Args:
            query (Optional[str], optional): The query to retrieve embeddings. Defaults to None.
            index_name (str, optional): The name of the index. Defaults to 'embeddings'.
            query_type (str, optional): The type of query to use. Defaults to 'match_all'.
            query_size (int, optional): The size of the query. Defaults to 1.
            query_params (Optional[dict], optional): Additional parameters for search. Defaults to None.

        Returns:
            List[dict]: A list of hits containing the retrieved embeddings.
        """
        query_params = query_params or {}
        if query == "default":
            query = cls.query
        if index_name is None:
            index_name = cls.index_name
        if any([index_name is None]):
            raise RuntimeError("Must provide index name!")
        if query is not None:
            if not isinstance(query, (bytes, str)):
                query = json.dumps(query)
        query_params.update({"size": query_size})
        if query is not None:
            raise NotImplementedError('Not implemented for this class.')
        if query is None:
            body = {
                "size": query_size,
                "sort": [{
                    "_seq_no": {
                        "order": "desc"
                    }
                }],
                "_source": ["embedding"],
            }
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response["hits"]["hits"]


class RandomEmbedGetter(EmbedGetter):

    @classmethod
    def retrieve_embeddings(
        cls,
        index_name: str = "embeddings",
        query_size: int = 1,
        **kwds
    ) -> List[dict]:
        body = {
            "size": query_size,
            "query": {
                "function_score": {
                    "query": {
                        "match_all": {}
                    },
                    "random_score": {}
                }
            },
            "_source": "embedding",
        }
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response["hits"]["hits"]


class CosineSimilarityEmbedGetter(EmbedGetter):

    @classmethod
    def retrieve_embeddings(
        cls,
        query: OpenSearchQuery | None = None,
        index_name: str = "embeddings",
        query_size: int = 1,
        **kwds
    ) -> List[dict]:
        body = {
            "size": query_size,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source":
                        "1.0 + cosineSimilarity(params.query_vector, doc['embedding'])",
                        "params": {
                            "query_vector": query
                        },
                    },
                }
            },
        }
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response["hits"]["hits"]


class FuzzyEmbedGetter(EmbedGetter):

    @classmethod
    def retrieve_embeddings(
        cls,
        query: str | None = None,
        index_name: str = "embeddings",
        query_size: int = 1,
        query_params: Optional[dict] = None,
    ) -> List[dict]:
        body = {
            "size": query_size,
            "query": {
                "fuzzy": {
                    "embedding": {
                        "value": query,
                        **(query_params or {}),
                    }
                }
            },
        }
        response = cls.opensearch_client.search(index=index_name, body=body)
        return response["hits"]["hits"]


@click.group()
def cli():
    pass


@cli.command()
def run_embedder(**kwds):
    """
    Run embedder to create embeddings.

    :return: The embeddings created by the embedder.
    """
    defaults = dict(grid_params=dict(num=8, kind="random"),
                    require_ssh_tunnel=False)
    kwds = {**defaults, **kwds}
    embedder = Embedder(**kwds)
    embedder.run()  # type: ignore


@cli.command()
@click.option("--query",
              default=None,
              help="The query to retrieve embeddings.")
@click.option("--index-name",
              default="embeddings",
              help="The name of the index.")
@click.option("--query-type",
              default="cosine",
              help="The type of query to use.")
@click.option("--query-size", default=1, help="The size of the query.")
@click.option("--query-params",
              default={'error_trace': True},
              help="Additional parameters for search.")
def retrieve_embeddings(**kwds):
    """
    Retrieve embeddings from opensearch.

    :param query: A string, or a list of strings, or a dictionary representing the query to retrieve embeddings for. Defaults to the sample text provided by the `Embedder` class.
    :return: A list of embeddings retrieved from opensearch.
    """
    if kwds.get("query") is None:
        resp = EmbedGetter().retrieve_embeddings(
            query=None, query_size=1)[0]
        print('Default query ID: ', resp["_id"])
        kwds["query"] = resp["_source"]["embedding"]
    #: retrieve embeddings from opensearch
    cls_dict = {
        "match_all": EmbedGetter,
        "random": RandomEmbedGetter,
        "cosine": CosineSimilarityEmbedGetter,
        "fuzzy": FuzzyEmbedGetter,
    }
    klass_ = cls_dict.get(kwds.pop("query_type"))
    if klass_ is None:
        raise ValueError(f"Invalid query type: {kwds['query_type']}")
    embeddings = klass_().retrieve_embeddings(**kwds)
    print("...done retrieving embeddings.")
    print(embeddings)
    return embeddings


if __name__ == "__main__":
    cli()
