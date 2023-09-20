import openai
import json
from opensearchpy import OpenSearch, helpers
import certifi
import uuid
import sys
from pathlib import Path
root_path = str(Path(__file__).parents[1])
mod_path = str(Path(__file__).parent)
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    sys.path.insert(0, str(Path(root_path).joinpath('runtime')))
if mod_path not in sys.path:
    sys.path.insert(0, mod_path)
from chalicelib.engine.cma_model_params import CMAModelParams
from chalicelib.engine.cma_sleepwake import CMASleepWakeModel
from chalicelib.secrets import get_secret
from sklearn.model_selection import ParameterGrid
import numpy as np


sample_text = '''
    {"_params": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "_DEFAULT_PARAMS_MODEL": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "_DEFAULT_PARAMS": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "t": [0.0, 1.0434782608695652, 2.0869565217391304, 3.1304347826086953, 4.173913043478261, 5.217391304347826, 6.260869565217391, 7.304347826086956, 8.347826086956522, 9.391304347826086, 10.434782608695652, 11.478260869565217, 12.521739130434781, 13.565217391304348, 14.608695652173912, 15.652173913043478, 16.695652173913043, 17.73913043478261, 18.782608695652172, 19.82608695652174, 20.869565217391305, 21.913043478260867, 22.956521739130434, 24.0], "tM": [7.0, 11.0, 17.5], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, "eps": 1e-18, "rng": null}
    '''


def create_parameter_search_grid():
    param_grid = {}
    cmap = CMAModelParams()
    bds = cmap.bounds.json()
    cdict = cmap.dict()
    param_grid = {k: np.linspace(bds['lb'][j], bds['ub'][j], num=5) for j, k in zip(range(len(cmap.__fields__)), cdict.get('bounded_param_keys'))}
    param_grid = ParameterGrid(param_grid)
    return param_grid


param_grid = create_parameter_search_grid()


host = 'localhost'
port = 9200
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
def get_embeddings(text):
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
