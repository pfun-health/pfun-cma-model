import openai
import json
from opensearchpy import OpenSearch, helpers
import certifi
from runtime.chalicelib.engine.cma_model_params import CMAModelParams

cmap = CMAModelParams()
bds = cmap.bounds.json()
from pandas import DataFrame
DataFrame(bds)


host = 'localhost'
port = 9200
auth = ('admin', 'admin') # For testing only. Don't store credentials in code.

# Provide a CA bundle if you use intermediate CAs with your root CA.
# If this is not given, the CA bundle is is discovered from the first available
# following options:
# - OpenSSL environment variables SSL_CERT_FILE and SSL_CERT_DIR
# - certifi bundle (https://pypi.org/project/certifi/)
# - default behavior of the connection backend (most likely system certs)
ca_certs_path = certifi.where()

# Optional client certificates if you don't want to use HTTP basic authentication.
# client_cert_path = '/full/path/to/client.pem'
# client_key_path = '/full/path/to/client-key.pem'

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
openai.api_key = "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"


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
    text = '''
    {"_params": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "_DEFAULT_PARAMS_MODEL": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "_DEFAULT_PARAMS": {"t": null, "N": 24, "d": 0.0, "taup": 1.0, "taug": 1.0, "B": 0.05, "Cm": 0.0, "toff": 0.0, "tM": [7.0, 11.0, 17.5], "seed": null, "eps": 1e-18, "lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "bounded_param_keys": ["d", "taup", "taug", "B", "Cm", "toff"], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}}, "t": [0.0, 1.0434782608695652, 2.0869565217391304, 3.1304347826086953, 4.173913043478261, 5.217391304347826, 6.260869565217391, 7.304347826086956, 8.347826086956522, 9.391304347826086, 10.434782608695652, 11.478260869565217, 12.521739130434781, 13.565217391304348, 14.608695652173912, 15.652173913043478, 16.695652173913043, 17.73913043478261, 18.782608695652172, 19.82608695652174, 20.869565217391305, 21.913043478260867, 22.956521739130434, 24.0], "tM": [7.0, 11.0, 17.5], "bounds": {"lb": [-12.0, 0.5, 0.1, 0.0, 0.0, -3.0], "ub": [14.0, 3.0, 3.0, 1.0, 2.0, 3.0], "keep_feasible": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, "eps": 1e-18, "rng": null}
    '''
    embedding = get_embeddings(text)
    print(embedding)
    doc_id = "embedding-0"
    return save_to_opensearch(embedding, doc_id)

if __name__ == "__main__":
    response = main()
    print(response)
