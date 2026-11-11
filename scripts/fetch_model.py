"""Download one pinned Diffusers snapshot, not duplicate standalone weights."""
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from flux_image_server.config import MODEL_ID, MODEL_REVISION


def main():
    destination = Path(os.environ.get("FLUX_MODEL_DIR", "/workspace/models/flux2-dev"))
    result = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_dir=destination,
        allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    (destination / "flux-snapshot.json").write_text(json.dumps({"repo": MODEL_ID, "revision": MODEL_REVISION}) + "\n")
    print("Downloaded pinned FLUX.2-dev Diffusers components to " + str(result))


if __name__ == "__main__":
    main()
