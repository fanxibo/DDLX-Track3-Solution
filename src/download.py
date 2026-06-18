import os
from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

def main():
    token = os.getenv("MODELSCOPE_TOKEN")

    if token:
        print("Logging in to ModelScope with token from environment variable...")
        api = HubApi()
        api.login(token)
    else:
        print("MODELSCOPE_TOKEN is not set. Trying public/anonymous download...")

    print("Downloading DDL_X dataset...")
    dataset_dir = snapshot_download(
        "DDLteam/DDL_X",
        repo_type="dataset",
        local_dir="./DDL_X_dataset"
    )
    print(f"Download completed. Dataset saved to: {dataset_dir}")

if __name__ == "__main__":
    main()
