import importlib.util
from pathlib import Path


def test_subset_excludes_duplicate_weights_and_unsafe_paths():
    path = Path(__file__).parents[1] / "scripts" / "fetch_model.py"
    spec = importlib.util.spec_from_file_location("fetch_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.wanted("transformer/diffusion_pytorch_model-00001-of-00007.safetensors")
    assert module.wanted("text_encoder/config.json")
    assert module.wanted("model_index.json")
    for name in ("flux2-dev.safetensors", "transformer/weights.bin", "../transformer/config.json", "/vae/config.json"):
        assert not module.wanted(name)
