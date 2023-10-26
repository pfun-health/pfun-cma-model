from graphql import build_ast_schema, parse, print_schema
from graphql.utilities import load_schema_from_file
import pfun_path_helper
import os
from pfun_cma_model.config import settings
from typing import Optional


def load_schema(schema_path: Optional[str] = None):
    schema_path = schema_path or settings.PFUN_APP_SCHEMA_PATH
    # Load the schema from a file
    schema_ast = load_schema_from_file("schema.graphql")
    # Build a schema object from the AST
    schema = build_ast_schema(schema_ast)
    return schema


def main():
    schema = load_schema()
    print(print_schema(schema))
    return schema


if __name__ == "__main__":
    schema = main()