import click
from typing import Union, List, Dict
import pfun_path_helper
from embed import Embedder, EmbedGetter


@click.group()
def cli():
    pass


@cli.command()
def run_embedder():
    """
    Run embedder to create embeddings.
    """
    embedder = Embedder()
    embeddings = embedder.run()
    print(embeddings[:5])


@cli.command()
@click.option('--query', '-q', default=Embedder.get_sample_text(), help='Query to retrieve embeddings.')
def retrieve_embeddings(query: Union[str, List[str], Dict]):
    """
    Retrieve embeddings from opensearch.
    """
    embeddings = EmbedGetter().retrieve_embeddings(query)
    print(embeddings[:5])


if __name__ == '__main__':
    cli()
