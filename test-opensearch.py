import os
import openai
import json
os.environ['SSL_CERT_FILE'] = '/etc/ssl/certs/server.pem'
from opensearchpy import OpenSearch, helpers
import ssl
import certifi



# Initialize OpenAI API
openai.api_key = "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"


# Initialize OpenSearch client
opensearch_client = OpenSearch(
    hosts=['node-0.example.com'],
    http_auth=('admin', 'admin'),
    scheme="http",
    port=9200,
    use_ssl=True,
    ca_certs=certifi.where()
)

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
    helpers.bulk(opensearch_client, [action])

# Main function
def main():
    text = "example text"
    embedding = get_embeddings(text)
    doc_id = "document_1"
    save_to_opensearch(embedding, doc_id)

if __name__ == "__main__":
    main()
