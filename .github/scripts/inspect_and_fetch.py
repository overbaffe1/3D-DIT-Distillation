import os
import sys
import json
import struct
import urllib.request
import zipfile
import subprocess
from pathlib import Path
from remotezip import RemoteZip

URLS = {
    "code": "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/3dgen-main.zip",
    "weights": "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/kandi3d_weights_aijc_kit.zip",
    "public_test": "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/objhq_splits_clean/public_test.zip",
    "train_001": "https://rndml-team-xr.obs.ru-moscow-1.hc.sbercloud.ru/mazurov/tmp_data/AIJC_data/objhq_splits_clean/train_archives/train_part_001.tar",
}

out_dir = Path("fetched_info")
out_dir.mkdir(exist_ok=True)

report = {"headers": {}, "code_files": [], "weights_files": [], "public_test_files": []}

for name, url in URLS.items():
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            report["headers"][name] = dict(resp.headers)
            print(f"{name}: {resp.headers.get('Content-Length')} bytes")
    except Exception as e:
        report["headers"][name] = {"error": str(e)}
        print(f"Error HEAD {name}: {e}")

# 1. Download 3dgen-main.zip
print("Downloading 3dgen-main.zip...")
urllib.request.urlretrieve(URLS["code"], "/tmp/3dgen-main.zip")
with zipfile.ZipFile("/tmp/3dgen-main.zip", "r") as zf:
    report["code_files"] = [
        {"filename": info.filename, "file_size": info.file_size}
        for info in zf.infolist()
    ]
    extract_dir = Path("baseline_repo")
    extract_dir.mkdir(exist_ok=True)
    zf.extractall(extract_dir)
    for p in extract_dir.rglob("*"):
        if p.is_file() and p.stat().st_size > 10 * 1024 * 1024:
            print(f"Removing large file from baseline_repo: {p} ({p.stat().st_size} bytes)")
            p.unlink()

# 2. Inspect public_test.zip via RemoteZip
print("Inspecting public_test.zip via RemoteZip...")
try:
    with RemoteZip(URLS["public_test"]) as zf:
        infolist = zf.infolist()
        report["public_test_total_entries"] = len(infolist)
        report["public_test_files"] = [
            {"filename": info.filename, "file_size": info.file_size}
            for info in infolist[:500]
        ]
        sample_extract = out_dir / "sample_test_data"
        sample_extract.mkdir(exist_ok=True)
        extracted_dinov3 = 0
        extracted_vox = 0
        for info in infolist:
            if info.filename.endswith("vox_idxi.npz") and extracted_vox < 3 and info.file_size < 5 * 1024 * 1024:
                zf.extract(info, sample_extract)
                extracted_vox += 1
            elif info.filename.endswith("_dinov3.npz") and extracted_dinov3 < 1 and info.file_size < 15 * 1024 * 1024:
                zf.extract(info, sample_extract)
                extracted_dinov3 += 1
except Exception as e:
    print(f"RemoteZip public_test error: {e}")
    report["public_test_error"] = str(e)

# 3. Inspect weights zip via RemoteZip
print("Inspecting weights zip via RemoteZip...")
try:
    with RemoteZip(URLS["weights"]) as zf:
        infolist = zf.infolist()
        report["weights_files"] = [
            {"filename": info.filename, "file_size": info.file_size}
            for info in infolist
        ]
        weights_meta_dir = out_dir / "weights_configs"
        weights_meta_dir.mkdir(exist_ok=True)
        for info in infolist:
            if any(info.filename.endswith(ext) for ext in [".json", ".yaml", ".yml", ".txt", ".py", ".md"]):
                if info.file_size < 5 * 1024 * 1024:
                    zf.extract(info, weights_meta_dir)
            elif info.filename.endswith(".safetensors"):
                with zf.open(info) as f:
                    header_len_bytes = f.read(8)
                    if len(header_len_bytes) == 8:
                        header_len = struct.unpack("<Q", header_len_bytes)[0]
                        print(f"Safetensors {info.filename}: header_len={header_len}")
                        if header_len < 10 * 1024 * 1024:
                            header_json = json.loads(f.read(header_len).decode("utf-8"))
                            safe_name = info.filename.replace("/", "__") + ".header.json"
                            with open(weights_meta_dir / safe_name, "w") as hf:
                                json.dump(header_json, hf, indent=2)
except Exception as e:
    print(f"RemoteZip weights error: {e}")
    report["weights_error"] = str(e)

with open(out_dir / "report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Done!")
