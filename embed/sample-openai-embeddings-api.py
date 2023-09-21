import openai
import json
from opensearchpy import OpenSearch, helpers
import certifi
import uuid
import subprocess
import threading
import requests
from sklearn.model_selection import ParameterGrid
import numpy as np
import pfun_path_helper as path_helper
from runtime.chalicelib.engine.cma_model_params import CMAModelParams
from runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from runtime.chalicelib.secrets import get_secret


class Code:
    def __init__(self):
        self.tunnel_thread = None
        self.sample_text = None
        self.completed_proc = None
        self.opensearch_client = None
        self.param_grid = None

    def setup_ssh_tunnel(self):
        def start_ssh_tunnel():
            self.completed_proc = subprocess.run(["ssh", "-fN", "-L", "9200:10.1.78.156:9200", "robbie@d2bd"], capture_output=True)
            if self.completed_proc.returncode != 0:
                print("Failed to start SSH tunnel.")
        self.tunnel_thread = threading.Thread(target=start_ssh_tunnel, daemon=True)
        self.tunnel_thread.start()
        print('Tunnel thread started')
        return self.tunnel_thread

    def get_sample_text(self):
        self.sample_text = requests.get('api.dev.pfun.app/api/params/default', headers={'Content-Type': 'application/json'}, timeout=5).json()
        return self.sample_text

    def create_parameter_search_grid(self):
        param_grid = {}
        cmap = CMAModelParams()
        bds = cmap.bounds.json()
        cdict = cmap.dict()
        param_grid = {k: np.linspace(bds['lb'][j], bds['ub'][j], num=5) for j, k in zip(range(len(cmap.__fields__)), cdict.get('bounded_param_keys'))}
        param_grid = ParameterGrid(param_grid)
        return param_grid

    def connect_opensearch(self):
        auth = ('admin', 'admin') # For testing only. Don't store credentials in code.
        ca_certs_path = certifi.where()

        # Create the client with SSL/TLS enabled, but hostname verification disabled.
        self.opensearch_client = OpenSearch(
            hosts = [{'host': 'node-0.example.com', 'port': 9200}],
            http_compress = True, # enables gzip compression for request bodies
            http_auth = auth,
            # client_cert = client_cert_path,
            # client_key = client_key_path,
            use_ssl = True,
            verify_certs = True,
            ssl_assert_hostname = False,
            ssl_show_warn = False,
            ca_certs = ca_certs_path
        )

    def get_embeddings(self, text) -> str:
        model = "text-embedding-ada-002"  # Replace with the model you want to use
        response = openai.Embedding.create(input=text, model=model)
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

    def main(self):
        embeddings = []
        print('Creating embeddings')
        for pset in self.param_grid:
            print('Creating embedding for ' + str(pset))
            raw_text = CMASleepWakeModel(**pset).json()
            embedding = self.get_embeddings(raw_text)
            doc_id = "embedding-" + str(uuid.uuid4())
            response = self.save_to_opensearch(embedding, doc_id)
            embeddings.append({'embedding': embedding, 'response': response})
            print('...Done creating embedding for ' + str(pset))
        print()
        print('...done.')
        return embeddings


if __name__ == "__main__":
    code = Code()
    code.tunnel_thread = code.setup_ssh_tunnel()
    code.sample_text = code.get_sample_text()
    code.param_grid = code.create_parameter_search_grid()
    code.connect_opensearch()
    embeddings = code.main()
    print(embeddings[:5])