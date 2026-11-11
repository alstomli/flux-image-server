from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def registry():
    path = Path(__file__).parents[1] / "scripts" / "registry_images.py"
    spec = importlib.util.spec_from_file_location("registry_images", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def content():
    manifest = {"config": {"digest": "sha256:" + "a" * 64, "size": 100, "mediaType": "config"},
                "layers": [{"digest": "sha256:" + value * 64, "size": 10, "mediaType": "gzip"} for value in ("b", "c")]}
    config = {"os": "linux", "architecture": "amd64", "rootfs": {"type": "layers", "diff_ids": ["first", "second"]},
              "config": {"Env": ["STATIC_EXAMPLE=private-value"], "Entrypoint": ["entrypoint"],
                         "Cmd": ["python", "-m", "flux_image_server"], "WorkingDir": "/app", "ExposedPorts": {"8001/tcp": {}}}}
    return manifest, config


def test_content_equivalence_excludes_environment_values_from_evidence(registry, content):
    evidence = registry.compare_images(*content, *deepcopy(content))
    assert evidence["environmentNames"] == ["STATIC_EXAMPLE"]
    assert "private-value" not in json.dumps(evidence)
    assert evidence["rootfsDiffIds"] == ["first", "second"]


@pytest.mark.parametrize("change", ["config", "layer_digest", "layer_size", "layer_media", "layer_order", "rootfs", "runtime"])
def test_changed_content_or_runtime_is_rejected(registry, content, change):
    manifest, config = deepcopy(content)
    if change == "config":
        manifest["config"]["digest"] = "sha256:" + "d" * 64
    elif change == "layer_digest":
        manifest["layers"][0]["digest"] = "sha256:" + "d" * 64
    elif change == "layer_size":
        manifest["layers"][0]["size"] += 1
    elif change == "layer_media":
        manifest["layers"][0]["mediaType"] = "different"
    elif change == "layer_order":
        manifest["layers"].reverse()
    elif change == "rootfs":
        config["rootfs"]["diff_ids"].reverse()
    else:
        config["config"]["Cmd"] = ["different"]
    with pytest.raises(registry.RegistryError):
        registry.compare_images(*content, manifest, config)


def test_public_repository_and_unused_tag_fail_closed(registry, monkeypatch):
    monkeypatch.setattr(registry, "hub_get", lambda path: (200, {"namespace": "zihan000", "name": "flux-inference", "is_private": True}))
    with pytest.raises(registry.RegistryError):
        registry.require_unused("1.0.0")
    monkeypatch.setattr(registry, "hub_get", lambda path: (200, {"namespace": "zihan000", "name": "flux-inference", "is_private": False}))
    with pytest.raises(registry.RegistryError, match="already exists"):
        registry.require_unused("1.0.0")
    monkeypatch.setattr(registry, "hub_get", lambda path: (503, None))
    with pytest.raises(registry.RegistryError):
        registry.require_unused("1.0.0")


@pytest.mark.parametrize("existing", [False, True])
def test_mirror_copies_all_content_without_clobber_or_verifies_existing(registry, content, monkeypatch, tmp_path, existing):
    calls = []
    monkeypatch.setattr(registry, "require_public_repo", lambda: None)
    monkeypatch.setattr(registry, "tag_exists", lambda _: existing)
    monkeypatch.setattr(registry, "digest", lambda ref, **_: ref.split("@", 1)[1] if ref.startswith(registry.GHCR_REPO) else "sha256:" + "e" * 64)
    monkeypatch.setattr(registry, "image", lambda *_args, **_kwargs: deepcopy(content))
    index = {"schemaVersion": 2, "manifests": [{"digest": "sha256:" + "f" * 64, "platform": {"os": "linux", "architecture": "amd64"}}]}
    def command(*args, **_):
        if args[0] == "manifest":
            return json.dumps(index)
        calls.append(args)
    monkeypatch.setattr(registry, "crane", command)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    registry.mirror("deployed-reference", tmp_path)
    assert len(calls) == (0 if existing else 1)
    if calls:
        assert calls[0][:2] == ("copy", "--no-clobber") and all("--platform" not in item for item in calls[0])
    evidence = json.loads((tmp_path / "deployed-reference.json").read_text())
    assert evidence["existingTagVerified"] is existing and evidence["anonymousPullVerified"]


def test_existing_mismatched_tag_is_never_overwritten(registry, content, monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "require_public_repo", lambda: None)
    monkeypatch.setattr(registry, "tag_exists", lambda _: True)
    monkeypatch.setattr(registry, "digest", lambda ref, **_: ref.split("@", 1)[1] if ref.startswith(registry.GHCR_REPO) else "sha256:" + "e" * 64)
    changed = deepcopy(content)
    changed[1]["config"]["Cmd"] = ["different"]
    monkeypatch.setattr(registry, "image", lambda ref, **_: deepcopy(content if ref.startswith(registry.GHCR_REPO) else changed))
    def no_write(*args, **_kwargs):
        if args[0] == "manifest":
            return json.dumps({"schemaVersion": 2, "manifests": []})
        raise AssertionError("Existing mismatched tags must not be overwritten")
    monkeypatch.setattr(registry, "crane", no_write)
    with pytest.raises(registry.RegistryError):
        registry.mirror("deployed-reference", tmp_path)


@pytest.mark.parametrize("reference", ["zihan000/flux-inference:latest",
    "docker.io/zihan000/flux-inference@sha256:" + "f" * 64,
    "ghcr.io/alstomli/flux-image-server@sha256:eb90674d975df24ab14732f49a86bb59777d4b5c3e1f13273eb9c980916d0585"])
def test_unverified_incremental_base_is_rejected(registry, tmp_path, reference):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM " + reference + "\n")
    with pytest.raises(registry.RegistryError):
        registry.verify_runtime_base(dockerfile)


def test_verified_incremental_base_only_contacts_dockerhub_anonymously(registry, monkeypatch, tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM " + registry.VERIFIED_RUNTIME_BASE + "\n")
    calls = []
    monkeypatch.setattr(registry, "require_public_repo", lambda: calls.append("public-target"))
    def inspect(reference, *, anonymous):
        assert reference == registry.VERIFIED_RUNTIME_BASE
        assert reference.startswith("docker.io/") and anonymous is True
        calls.append(reference)
        return reference.split("@", 1)[1]
    monkeypatch.setattr(registry, "digest", inspect)
    monkeypatch.setattr(registry, "image", inspect)
    registry.verify_runtime_base(dockerfile)
    assert calls == ["public-target", registry.VERIFIED_RUNTIME_BASE, registry.VERIFIED_RUNTIME_BASE]


def test_index_verification_preserves_platforms_and_attestations(registry):
    index = {"schemaVersion": 2, "manifests": [{"digest": "image", "platform": {"os": "linux", "architecture": "amd64"}},
                                               {"digest": "attestation", "platform": {"os": "unknown", "architecture": "unknown"}}]}
    registry.compare_indexes(index, deepcopy(index))
    for entries in (index["manifests"][:1], [*index["manifests"], {"digest": "unexpected"}]):
        with pytest.raises(registry.RegistryError):
            registry.compare_indexes(index, {**index, "manifests": entries})


def test_anonymous_inspection_uses_empty_temporary_auth_and_redacts_failures(registry, monkeypatch, tmp_path):
    monkeypatch.setattr(registry.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("DOCKER_CONFIG", "logged-in-directory")
    monkeypatch.setenv("REGISTRY_AUTH_FILE", "logged-in-file")
    paths = []
    def inspect(_command, **options):
        environment = options["env"]
        assert environment["DOCKER_CONFIG"] != "logged-in-directory"
        configuration = Path(environment["REGISTRY_AUTH_FILE"])
        assert configuration.parent == Path(environment["DOCKER_CONFIG"])
        assert json.loads(configuration.read_text()) == {"auths": {}}
        paths.append(configuration.parent)
        return SimpleNamespace(returncode=1, stdout="", stderr="credential-should-never-appear")
    monkeypatch.setattr(registry.subprocess, "run", inspect)
    with pytest.raises(registry.RegistryError) as result:
        registry.crane("digest", "public-image", anonymous=True)
    assert "credential-should-never-appear" not in str(result.value)
    assert paths and not paths[0].exists()
    assert registry.os.environ["DOCKER_CONFIG"] == "logged-in-directory"
