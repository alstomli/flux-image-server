"""Stage one verified checkpoint file locally, then whole-copy to object storage.

HF download/cache locks never touch the global volume. Only the final component
files and completed snapshot manifest are written to FLUX_MODEL_DIR.
"""
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

from flux_image_server.config import MODEL_ID, MODEL_REVISION

COMPONENTS = {"scheduler", "tokenizer", "text_encoder", "transformer", "vae"}


def wanted(name):
    path = PurePosixPath(name)
    return (not path.is_absolute() and ".." not in path.parts
            and (name == "model_index.json" or path.parts[0] in COMPONENTS)
            and path.suffix not in (".bin", ".msgpack", ".h5"))


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    destination = Path(os.environ.get("FLUX_MODEL_DIR", "/workspace/models/flux2-dev"))
    # These paths must be local, not paths under the global/object volume.
    stage_root = Path(os.environ.get("FLUX_DOWNLOAD_STAGE", tempfile.gettempdir()))
    marker_root = Path(os.environ.get("FLUX_DOWNLOAD_STATE", "/opt/flux-download-state")) / MODEL_REVISION
    destination.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)
    marker_root.mkdir(parents=True, exist_ok=True)
    manifest = destination / "flux-snapshot.json"
    # No concurrent service should load a partially replaced snapshot.
    manifest.unlink(missing_ok=True)
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("Model metadata does not match the pinned revision")
    files = [file for file in info.siblings if wanted(file.rfilename)]
    total = sum(file.size or 0 for file in files)
    completed = 0
    print(json.dumps({"phase": "download", "files": len(files), "totalBytes": total}), flush=True)
    for file in files:
        output = destination / file.rfilename
        output.parent.mkdir(parents=True, exist_ok=True)
        expected = getattr(file.lfs, "sha256", None) if file.lfs else None
        if isinstance(file.lfs, dict):
            expected = file.lfs.get("sha256")
        marker = marker_root / (sha256(file.rfilename.encode()).hexdigest() + ".json")
        expected_marker = {"revision": MODEL_REVISION, "size": file.size, "sha256": expected}
        if output.exists() and file.size is not None and output.stat().st_size == file.size and marker.exists():
            try:
                if json.loads(marker.read_text()) == expected_marker:
                    completed += file.size
                    continue
            except (OSError, ValueError):
                pass
        for attempt in range(4):
            try:
                # Deleting this per-file local cache after each copy keeps staging
                # space bounded by the largest shard, rather than the full model.
                with tempfile.TemporaryDirectory(prefix="flux-checkpoint-", dir=stage_root) as cache:
                    source = Path(hf_hub_download(MODEL_ID, file.rfilename, revision=MODEL_REVISION,
                                                 token=token, cache_dir=cache))
                    if file.size is not None and source.stat().st_size != file.size:
                        raise RuntimeError("Checkpoint size mismatch")
                    if expected and digest(source) != expected:
                        raise RuntimeError("Checkpoint SHA256 mismatch")
                    with source.open("rb") as reader, output.open("wb") as writer:
                        shutil.copyfileobj(reader, writer, 8 * 1024**2)
                    if output.stat().st_size != source.stat().st_size:
                        raise RuntimeError("Object-volume copy size mismatch")
                marker.write_text(json.dumps(expected_marker))
                completed += file.size or output.stat().st_size
                print(json.dumps({"file": file.rfilename, "completedBytes": completed,
                                  "totalBytes": total, "sha256Verified": bool(expected)}), flush=True)
                break
            except Exception as error:
                # Library exceptions can contain authenticated URLs; print type only.
                print(json.dumps({"file": file.rfilename, "attempt": attempt + 1,
                                  "errorClass": type(error).__name__}), flush=True)
                if attempt == 3:
                    raise RuntimeError("Pinned checkpoint download failed; inspect sanitized progress") from None
                time.sleep(3 * (attempt + 1))
    manifest.write_text(json.dumps({"repo": MODEL_ID, "revision": MODEL_REVISION}) + "\n")
    print(json.dumps({"phase": "complete", "completedBytes": completed}), flush=True)


if __name__ == "__main__":
    main()
