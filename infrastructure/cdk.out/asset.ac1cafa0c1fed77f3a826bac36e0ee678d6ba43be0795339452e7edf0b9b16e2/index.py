import subprocess
import shutil
import os


def handler(event, context):
    # Generate the Chalice SDK
    subprocess.run(["chalice", "generate-sdk", "/tmp/sdk"], check=True)
    subprocess.run(["zip", "-r", "/tmp/sdk.zip", "/tmp/sdk"], check=True)
    sdk_zipfile = "/tmp/sdk.zip"
    with open(sdk_zipfile, "rb") as f:
        sdk = f.read()

    return {
        "SDK": sdk
    }