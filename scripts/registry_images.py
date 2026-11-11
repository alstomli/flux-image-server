"""Content-preserving registry mirroring and fail-closed Docker Hub publishing.

Registry credentials come only from Docker's normal login configuration. They are
never command arguments or captured in evidence. Anonymous verification uses a
separate empty Docker configuration.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DOCKERHUB_REPO = "zihan000/flux-inference"
DESTINATION = "docker.io/" + DOCKERHUB_REPO
# Independently verified mirror: index, every child, config, layers and rootfs.
# Future builds trust this immutable pin without contacting the bootstrap registry.
VERIFIED_RUNTIME_BASE = DESTINATION + "@sha256:eb90674d975df24ab14732f49a86bb59777d4b5c3e1f13273eb9c980916d0585"
GHCR_REPO = "ghcr.io/alstomli/flux-image-server"
MIRRORS = {
    "deployed-reference": {
        "source": GHCR_REPO + "@sha256:3a0319d671f400708670d33287ff2704f357c7763c3843d886b9fabc4b35645a",
        "tag": "sha-4c867b632b074c7f1352c7fae058c4d159ede043",
    },
    "prior-runtime": {
        "source": GHCR_REPO + "@sha256:eb90674d975df24ab14732f49a86bb59777d4b5c3e1f13273eb9c980916d0585",
        "tag": "runtime-eb90674d975d",
    },
}


class RegistryError(RuntimeError):
    pass


def tag_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", value):
        raise RegistryError("Invalid immutable image tag")
    return value


def hub_get(path):
    request = Request("https://hub.docker.com/v2/repositories/" + DOCKERHUB_REPO + path,
                      headers={"Accept": "application/json", "User-Agent": "flux-registry-verification"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return 404, None
        raise RegistryError(f"Docker Hub lookup failed with HTTP {error.code}") from None


def require_public_repo():
    status, repository = hub_get("")
    if status != 200 or not repository:
        raise RegistryError("Create public Docker Hub repository zihan000/flux-inference first")
    namespace = repository.get("namespace") or repository.get("user")
    if namespace != "zihan000" or repository.get("name") != "flux-inference" or repository.get("is_private") is not False:
        raise RegistryError("Expected the exact public Docker Hub repository zihan000/flux-inference")


def tag_exists(tag):
    status, _ = hub_get("/tags/" + tag_name(tag))
    if status not in (200, 404):
        raise RegistryError("Cannot establish whether the Docker Hub tag exists")
    return status == 200


def require_unused(tag):
    require_public_repo()
    if tag_exists(tag):
        raise RegistryError("The immutable destination tag already exists; choose a new tag")


def crane(*arguments, anonymous=False):
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="flux-anonymous-registry-") as temporary:
        if anonymous:
            configuration = Path(temporary) / "config.json"
            configuration.write_text('{"auths":{}}\n')
            environment["DOCKER_CONFIG"] = temporary
            environment["REGISTRY_AUTH_FILE"] = str(configuration)
        result = subprocess.run([os.environ.get("CRANE", "crane"), *arguments], capture_output=True,
                                text=True, env=environment)
    if result.returncode:
        # Keep credential helper details and registry response internals out of logs.
        raise RegistryError("Registry command failed: " + arguments[0])
    return result.stdout.strip()


def digest(reference, anonymous=False):
    value = crane("digest", reference, anonymous=anonymous)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise RegistryError("Expected a SHA256 registry digest")
    return value


def image(reference, anonymous=False):
    manifest = json.loads(crane("--platform=linux/amd64", "manifest", reference, anonymous=anonymous))
    config = json.loads(crane("--platform=linux/amd64", "config", reference, anonymous=anonymous))
    if config.get("os") != "linux" or config.get("architecture") != "amd64":
        raise RegistryError("The mirrored runtime must be linux/amd64")
    return manifest, config


def descriptor(value):
    return {name: value[name] for name in ("digest", "size", "mediaType")}


def compare_images(source_manifest, source_config, target_manifest, target_config):
    if descriptor(source_manifest["config"]) != descriptor(target_manifest["config"]):
        raise RegistryError("Mirrored config descriptor differs")
    source_layers = [descriptor(layer) for layer in source_manifest["layers"]]
    target_layers = [descriptor(layer) for layer in target_manifest["layers"]]
    if source_layers != target_layers:
        raise RegistryError("Mirrored ordered compressed layer descriptors differ")
    for key in ("os", "architecture", "rootfs", "config"):
        if source_config.get(key) != target_config.get(key):
            raise RegistryError("Mirrored runtime configuration differs: " + key)
    canonical = json.dumps(source_config["config"], sort_keys=True, separators=(",", ":")).encode()
    return {"configDigest": source_manifest["config"]["digest"], "layers": source_layers,
            "rootfsDiffIds": source_config["rootfs"]["diff_ids"], "runtimeConfigSha256": sha256(canonical).hexdigest(),
            "runtime": {key: source_config["config"].get(key) for key in ("Entrypoint", "Cmd", "WorkingDir", "ExposedPorts")},
            "environmentNames": [entry.split("=", 1)[0] for entry in source_config["config"].get("Env", [])]}


def compare_indexes(source, target):
    # These fixed migration sources are OCI indexes including amd64 images and
    # attestations. A registry may reserialize an index, but may not drop, add,
    # rewrite, or reorder the meaning of its child descriptors/platforms.
    source_entries, target_entries = source.get("manifests"), target.get("manifests")
    if not isinstance(source_entries, list) or not isinstance(target_entries, list):
        raise RegistryError("Expected complete source and destination image indexes")
    canonical = lambda entries: sorted(json.dumps(entry, sort_keys=True, separators=(",", ":")) for entry in entries)
    if canonical(source_entries) != canonical(target_entries):
        raise RegistryError("Mirrored index child descriptors or platforms differ")
    for key in ("schemaVersion", "annotations", "subject", "artifactType"):
        if source.get(key) != target.get(key):
            raise RegistryError("Mirrored index metadata differs: " + key)


def mirror(artifact, evidence_dir):
    specification = MIRRORS[artifact]
    source, tag = specification["source"], specification["tag"]
    destination = DESTINATION + ":" + tag
    require_public_repo()
    source_digest = digest(source, anonymous=True)
    if source_digest != source.split("@", 1)[1]:
        raise RegistryError("Pinned source digest mismatch")
    source_manifest, source_config = image(source, anonymous=True)
    source_index_raw = crane("manifest", source, anonymous=True)
    existed = tag_exists(tag)
    if not existed:
        # Copy all manifests/attestations. Do not select a platform during copy,
        # because that would intentionally rewrite the original OCI index.
        crane("copy", "--no-clobber", source, destination)
    destination_digest = digest(destination, anonymous=True)
    immutable_destination = DESTINATION + "@" + destination_digest
    target_manifest, target_config = image(immutable_destination, anonymous=True)
    target_index_raw = crane("manifest", immutable_destination, anonymous=True)
    compare_indexes(json.loads(source_index_raw), json.loads(target_index_raw))
    evidence = compare_images(source_manifest, source_config, target_manifest, target_config)
    evidence.update(artifact=artifact, source=source, destination=immutable_destination, tag=tag,
                    sourceDigest=source_digest, destinationDigest=destination_digest,
                    anonymousPullVerified=True, existingTagVerified=existed,
                    sourceIndexRaw=source_index_raw, destinationIndexRaw=target_index_raw)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / (artifact + ".json")
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"artifact": artifact, "destination": immutable_destination,
                      "contentVerified": True, "anonymousPullVerified": True}))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(f"\n{artifact}: `{immutable_destination}` — platform/config/ordered layers/rootfs verified; anonymous pull verified.\n")


def verify_runtime_base(dockerfile):
    lines = [line.strip() for line in dockerfile.read_text().splitlines() if line.strip().startswith("FROM ")]
    if len(lines) != 1:
        raise RegistryError("Expected one explicitly pinned incremental runtime base")
    reference = lines[0].split()[1]
    if reference != VERIFIED_RUNTIME_BASE:
        raise RegistryError("Incremental base must use the verified immutable Docker Hub runtime mirror")
    require_public_repo()
    if digest(reference, anonymous=True) != reference.split("@", 1)[1]:
        raise RegistryError("Incremental Docker Hub base digest mismatch")
    image(reference, anonymous=True)
    print("Pinned Docker Hub runtime digest, linux/amd64 platform and anonymous pull verified")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    unused = subparsers.add_parser("require-unused")
    unused.add_argument("tag")
    check = subparsers.add_parser("verify-public")
    check.add_argument("reference")
    base = subparsers.add_parser("verify-runtime-base")
    base.add_argument("dockerfile", type=Path)
    copying = subparsers.add_parser("mirror")
    copying.add_argument("artifact", choices=["both", *MIRRORS])
    copying.add_argument("--evidence-dir", type=Path, default=Path("registry-evidence"))
    args = parser.parse_args()
    if args.operation == "require-unused":
        require_unused(args.tag)
    elif args.operation == "verify-public":
        require_public_repo()
        if not args.reference.startswith(DESTINATION + "@sha256:"):
            raise RegistryError("Expected an immutable Docker Hub deployment reference")
        actual = digest(args.reference, anonymous=True)
        if actual != args.reference.split("@", 1)[1]:
            raise RegistryError("Anonymous destination digest mismatch")
        image(args.reference, anonymous=True)
        print(json.dumps({"reference": args.reference, "anonymousPullVerified": True}))
    elif args.operation == "verify-runtime-base":
        verify_runtime_base(args.dockerfile)
    else:
        for artifact in MIRRORS if args.artifact == "both" else [args.artifact]:
            mirror(artifact, args.evidence_dir)


if __name__ == "__main__":
    main()
