import os
import sys
import shutil
import urllib.request
import zipfile
from pathlib import Path

CODE_URL = "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/3dgen-main.zip"

print("Downloading 3dgen-main.zip...")
urllib.request.urlretrieve(CODE_URL, "/tmp/3dgen-main.zip")
extract_dir = Path("baseline_repo")
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir(exist_ok=True)

with zipfile.ZipFile("/tmp/3dgen-main.zip", "r") as zf:
    zf.extractall(extract_dir)

# Remove .git directory inside 3dgen-main so git adds the actual files instead of a submodule
git_dir = extract_dir / "3dgen-main" / ".git"
if git_dir.exists():
    shutil.rmtree(git_dir)

for p in extract_dir.rglob("*"):
    if p.is_file() and p.stat().st_size > 10 * 1024 * 1024:
        print(f"Removing large file from baseline_repo: {p} ({p.stat().st_size} bytes)")
        p.unlink()

print("Extracted files:")
for p in sorted(extract_dir.rglob("*")):
    if p.is_file():
        print(f"  {p} ({p.stat().st_size} B)")
