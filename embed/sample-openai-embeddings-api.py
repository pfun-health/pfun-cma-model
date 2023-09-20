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
import path_helper
from runtime.chalicelib.engine.cma_model_params import CMAModelParams
from runtime.chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from runtime.chalicelib.secrets import get_secret

tunnel_thread = None
sample_text = None
completed_proc = None

def setup_ssh_tunnel():
    def start_ssh_tunnel():
        completed_proc = subprocess.run(["ssh", "-fN", "-L", "9200:10.1.78.156:9200", "robbie@d2bd"], capture_output=True)
        if completed_proc.returncode != 0:
            print("Failed to start SSH tunnel.")
    tunnel_thread = threading.Thread(target=start_ssh_tunnel, daemon=True)
    tunnel_thread.start()
    print('Tunnel thread started')
    return tunnel_thread

tunnel_thread = setup_ssh_tunnel()

def get_sample_text():
    global sample_text
    sample_text = requests.get('api.dev.pfun.app/api/params/default', headers={'Content-Type': 'application/json'}, timeout=5).json()
    return sample_text

def create_parameter_search_grid():
    param_grid = {}
    cmap = CMAModelParams()
    bds = cmap.bounds.json()
    cdict = cmap.dict()
    param_grid = {k: np.linspace(bds['lb'][j], bds['ub'][j], num=5) for j, k in zip(range(len(cmap.__fields__)), cdict.get('bounded_param_keys'))}
    param_grid = ParameterGrid(param_grid)
    return param_grid

param_grid = create_parameter_search_grid()

opensearch_client = None

def connect_opensearch():
    global opensearch_client
    auth = ('admin', 'admin') # For testing only. Don't store credentials in code.
    ca_certs_path = certifi.where()

    # Create the client with SSL/TLS enabled, but hostname verification disabled.
    opensearch_client = OpenSearch(
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

# Initialize OpenAI API
openai.api_key = get_secret('openai-api-key-emacs', region_name='us-east-1')

# Function to get embeddings from OpenAI
def get_embeddings(text) -> str:
    model = "text-embedding-ada-002"  # Replace with the model you want to use
    response = openai.Embedding.create(input=text, model=model)
    return response.get('data')

# Function to save embeddings to OpenSearch
def save_to_opensearch(embedding, doc_id):
    action = {
        "_op_type": "index",
        "_index": "embeddings",
        "_id": doc_id,
        "embedding": embedding
    }
    response = helpers.bulk(opensearch_client, [action])
    return response

# Main function
def main():
    embeddings = []
    print('Creating embeddings')
    # Parameter grid for search
    for pset in param_grid:
        print('Creating embedding for ' + str(pset))
        raw_text = CMASleepWakeModel(**pset).json()
        embedding = get_embeddings(raw_text)
        doc_id = "embedding-" + str(uuid.uuid4())
        response = save_to_opensearch(embedding, doc_id)
        embeddings.append({'embedding': embedding, 'response': response})
        print('...Done creating embedding for ' + str(pset))
    print()
    print('...done.')
    return embeddings

if __name__ == "__main__":
    embeddings = main()
    print(embeddings[:5])