import boto3
import json
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


client = boto3.client('lambda')


def fix_lambda_layer_version():
    """
    Fix the lambda layer version.

    This function retrieves the latest version of the lambda layer 'pfun-cma-model-deps-dev-managed-layer'
    and updates the configuration file with the latest version.

    Parameters:
    None

    Returns:
    None
    """
    print('fixing lambda layer version...')
    versions = client.list_layer_versions(
        LayerName='pfun-cma-model-deps-dev-managed-layer')['LayerVersions']
    #: get latest version
    latest_version_arn = max(
        versions, key=lambda x: pd.Timestamp(x['CreatedDate']).timestamp())
    print('latest_version info:\n', latest_version_arn, '\n')
    latest_version_arn = latest_version_arn['LayerVersionArn']
    config_fpath = Path(__file__).parents[1].joinpath(
        '.chalice', 'config.json').resolve()
    config_dict = json.loads(config_fpath.read_text(encoding='utf-8'))
    config_dict['layers'] = [latest_version_arn, ]
    json.dump(config_dict, config_fpath.open('w', encoding='utf-8'), indent=4)
    print('fixed config:\n', json.dumps(config_dict, indent=4), '\n')
    print('...fixed lambda layer version.')
    print()


if __name__ == '__main__':
    fix_lambda_layer_version()
