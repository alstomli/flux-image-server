"""Bounded, local-only reference decoding; no image metadata leaves this module."""
from dataclasses import dataclass
import base64
import binascii
from hashlib import sha256
from io import BytesIO
import warnings

from PIL import Image, UnidentifiedImageError

from .errors import ServiceError

MAX_REFERENCE_BYTES = 3 * 1024**2
MAX_EDIT_REQUEST_BYTES = 4 * 1024**2 + 16 * 1024
SIZE = (1024, 1024)
PREFIXES = {"data:image/png;base64,": "PNG", "data:image/jpeg;base64,": "JPEG"}


def invalid():
    return ServiceError("invalid_reference", "Use one valid 1024 × 1024 PNG or JPEG reference image up to 3 MiB.", 400)


def decode_rgb(content: bytes, expected_format: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content), formats=["PNG", "JPEG"]) as image:
                if image.format != expected_format or image.size != SIZE:
                    raise invalid()
                if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                    raise invalid()
                image.verify()
            # Verify and full decode are separate operations in Pillow. Check size
            # before loading pixels and strip metadata by creating a fresh RGB image.
            with Image.open(BytesIO(content), formats=["PNG", "JPEG"]) as image:
                image.load()
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    result = Image.new("RGB", SIZE, "white")
                    result.paste(rgba, mask=rgba.getchannel("A"))
                    rgba.close()
                else:
                    result = image.convert("RGB")
                result.info.clear()
                return result
    except ServiceError:
        raise
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise invalid() from None


@dataclass(frozen=True)
class ReferenceImage:
    content: bytes
    format: str
    fingerprint: str

    def open_rgb(self):
        return decode_rgb(self.content, self.format)


def validate_reference(value):
    if not isinstance(value, str):
        raise invalid()
    prefix = next((prefix for prefix in PREFIXES if value.startswith(prefix)), None)
    if prefix is None:
        raise invalid()
    encoded = value[len(prefix):]
    if not encoded or len(encoded) > 4 * ((MAX_REFERENCE_BYTES + 2) // 3):
        raise ServiceError("reference_too_large", "The reference image exceeds 3 MiB.", 413)
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise invalid() from None
    if not content or len(content) > MAX_REFERENCE_BYTES:
        raise ServiceError("reference_too_large", "The reference image exceeds 3 MiB.", 413)
    if base64.b64encode(content).decode("ascii") != encoded:
        raise invalid()
    expected = PREFIXES[prefix]
    if expected == "PNG" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise invalid()
    if expected == "JPEG" and not content.startswith(b"\xff\xd8\xff"):
        raise invalid()
    image = decode_rgb(content, expected)
    image.close()
    return ReferenceImage(content, expected, sha256(content).hexdigest())
