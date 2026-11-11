# Docker Hub publishing and exact-image migration

The public image repository is `docker.io/zihan000/flux-inference`. The GitHub
repository remains `alstomli/flux-image-server`. No BF16 placement, no-offload
policy, queue/admission behavior, model revision, ports, or runtime environment is
changed by registry migration.

## Credentials and publication

Configure `DOCKER_USERNAME` and `DOCKER_PASSWORD` in this repository's
[Actions secrets](https://github.com/alstomli/flux-image-server/settings/secrets/actions).
`DOCKER_PASSWORD` is a Docker Hub write access token; keep its value in GitHub's
secret store rather than chat, source, workflow inputs, or build arguments. This
matches the existing Qwen publishing convention. Secrets in the Qwen repository
are encrypted and are neither read nor copied by this project.

The repository must already exist publicly. Every publisher checks the exact
public target and fails closed if an immutable tag exists or its state cannot be
established. Candidates use `sha-<full Git commit>`; manual main-branch releases
use a new `major.minor.patch` tag. No movable `latest` tag is created. Existing
GHCR images are retained; future candidate/release workflows publish Docker Hub
only and need no GitHub Packages write permission.

## Initial copy, not rebuild

`mirror.yml` copies these fixed, previously verified images:

| Artifact | Immutable GHCR source digest | Docker Hub tag |
| --- | --- | --- |
| Deployed reference-editing image | `sha256:3a0319d671f400708670d33287ff2704f357c7763c3843d886b9fabc4b35645a` | `sha-4c867b632b074c7f1352c7fae058c4d159ede043` |
| Prior GPU-verified runtime base | `sha256:eb90674d975df24ab14732f49a86bb59777d4b5c3e1f13273eb9c980916d0585` | `runtime-eb90674d975d` |

Pushes to the exact migration branch register the workflow and run read-only
preflight. They do not perform a registry copy. The operator explicitly dispatches
the mirror workflow with `artifact: both` (or one named artifact) after credentials
are configured. The workflow permits only main or the trusted migration branch.
If a new workflow cannot be dispatched from its feature ref, review and merge
the workflow PR first; this does not change the running Pod.

The pinned official `crane v0.22.1` binary is downloaded with its reviewed SHA256:
`0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0`.
`crane copy --no-clobber` copies all index children, including attestations, without
running apt, pip, model download, or a container build. The source references are
fixed digests rather than mutable input tags.

An existing destination tag is accepted only after full matching-content
verification; it is never overwritten. A mismatched existing tag fails the job.
This makes a partially completed two-image migration safe to rerun.

## Required evidence

The mirror helper records source/destination index digests and raw index JSON.
It verifies all child descriptors/platforms and index metadata, then independently
compares the selected `linux/amd64` image's config descriptor, ordered compressed
layer digests/sizes/media types, rootfs diff IDs, and full runtime configuration.
Registry index serialization may differ; image/runtime content may not.

Anonymous post-copy verification uses an empty temporary Docker authentication
configuration, distinct from the write login. Detailed evidence appears in the
Actions summary with the exact destination deployment reference. Environment
values and credentials are omitted from evidence; only environment names and a
runtime configuration hash are recorded.

The operator independently checks the destination digest and evidence before
changing `Dockerfile.incremental` to the mirrored prior-runtime digest, or changing
the RunPod image to the mirrored deployed-image digest. The candidate workflow
then verifies that the Docker Hub incremental base still matches the known runtime
before a source-only build. The ordinary pinned `Dockerfile` is preserved.

The repository does not claim Docker Hub deployment merely because a workflow or
PR exists. Publication requires successful copy/build and anonymous verification;
RunPod rollout is a separate operator action, followed by readiness, BF16 device
placement, memory, image editing, and metrics checks. Keep the prior immutable
GHCR deployment reference as rollback until those checks pass.

References: [official crane copy documentation](https://github.com/google/go-containerregistry/blob/v0.22.1/cmd/crane/doc/crane_copy.md),
[official release](https://github.com/google/go-containerregistry/releases/tag/v0.22.1).
