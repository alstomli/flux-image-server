"""Install the reviewed official crane binary into a caller-selected tool folder."""
import argparse
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import platform
import tarfile
from urllib.request import urlopen

VERSION = "v0.22.1"
ARCHIVE = "go-containerregistry_Linux_x86_64.tar.gz"
CHECKSUM = "0ab7a1d6932a213aed964ce97666c3077fe691c8606413674a8b3e0b9ec4cda0"
URL = f"https://github.com/google/go-containerregistry/releases/download/{VERSION}/{ARCHIVE}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("The pinned registry-copy tool requires Linux amd64")
    with urlopen(URL, timeout=60) as response:
        data = response.read(32 * 1024**2 + 1)
    if len(data) > 32 * 1024**2 or sha256(data).hexdigest() != CHECKSUM:
        raise SystemExit("Official crane archive checksum mismatch")
    with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as archive:
        member = archive.getmember("crane")
        if not member.isfile():
            raise SystemExit("Expected a regular crane executable")
        stream = archive.extractfile(member)
        if stream is None:
            raise SystemExit("Missing crane executable")
        args.destination.mkdir(parents=True, exist_ok=True)
        executable = args.destination / "crane"
        executable.write_bytes(stream.read())
        executable.chmod(0o755)
    print(f"Installed official crane {VERSION}; archive SHA256 verified")


if __name__ == "__main__":
    main()
