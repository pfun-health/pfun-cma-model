import click
from typing import Union, List, Dict
import pfun_path_helper
import os
import runpy
globals().update(runpy.run_path(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'embed.py')))


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
@click.option('--query', '-q', default=None, help='Query to retrieve embeddings.')
def retrieve_embeddings(query: Union[str, List[str], Dict]):
    """
    Retrieve embeddings from opensearch.
    """
    embeddings = EmbedGetter().retrieve_embeddings(query)
    print(embeddings[:5])


if __name__ == '__main__':
    cli()
