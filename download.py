import requests
import tarfile
import os
from tqdm import tqdm

def download_with_progress(url: str, dest_path: str, chunk_size: int = 1024 * 1024) -> None:
    """Download a file showing a progress bar."""
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        desc = os.path.basename(dest_path)
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {desc}"
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive chunks
                    f.write(chunk)
                    bar.update(len(chunk))

def extract_tar_with_progress(tar_path: str, out_dir: str = ".") -> None:
    """Extract a .tar.gz while showing per-file progress."""
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), unit="file", desc=f"Extracting {os.path.basename(tar_path)}") as bar:
            for m in members:
                tar.extract(m, path=out_dir)  # for extra safety, validate paths in production
                bar.update(1)