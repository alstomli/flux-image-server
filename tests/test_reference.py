import base64
from io import BytesIO
import struct
import zlib

from PIL import Image
import pytest

from flux_image_server.errors import ServiceError
from flux_image_server.reference import validate_reference, MAX_REFERENCE_BYTES


def image_url(format="PNG", color="red", mode="RGB", size=(1024, 1024)):
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format=format)
    mime = "image/png" if format == "PNG" else "image/jpeg"
    return "data:" + mime + ";base64," + base64.b64encode(output.getvalue()).decode()


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_valid_reference_fully_decodes_and_strips_metadata(format):
    reference = validate_reference(image_url(format))
    with reference.open_rgb() as decoded:
        assert decoded.size == (1024, 1024) and decoded.mode == "RGB" and decoded.info == {}


def test_transparency_is_composited_on_white():
    reference = validate_reference(image_url(mode="RGBA", color=(255, 0, 0, 0)))
    with reference.open_rgb() as decoded:
        assert decoded.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("value", [None, [], "https://example.com/image.png", "AAAA", "data:image/svg+xml;base64,AAAA",
                                  "data:image/png;base64,%%%%", "data:image/png;base64,AA==", "data:image/jpeg;base64,iVBORw0KGgo="])
def test_reference_protocol_and_decode_rejections(value):
    with pytest.raises(ServiceError) as result:
        validate_reference(value)
    assert result.value.status == 400


def test_oversized_binary_is_rejected_before_image_decode():
    data = b"\x89PNG\r\n\x1a\n" + b"x" * MAX_REFERENCE_BYTES
    with pytest.raises(ServiceError) as result:
        validate_reference("data:image/png;base64," + base64.b64encode(data).decode())
    assert result.value.status == 413


def test_wrong_dimensions_and_corrupt_pixels_are_rejected():
    with pytest.raises(ServiceError):
        validate_reference(image_url(size=(1024, 1023)))
    url = image_url()
    content = base64.b64decode(url.split(",", 1)[1])
    idat = content.index(b"IDAT")
    damaged = bytearray(content)
    damaged[idat + 10] ^= 0xFF
    with pytest.raises(ServiceError):
        validate_reference("data:image/png;base64," + base64.b64encode(damaged).decode())


def test_png_decompression_bomb_header_is_rejected():
    content = bytearray(base64.b64decode(image_url().split(",", 1)[1]))
    content[16:24] = struct.pack(">II", 100000, 100000)
    content[29:33] = struct.pack(">I", zlib.crc32(content[12:29]))
    with pytest.raises(ServiceError):
        validate_reference("data:image/png;base64," + base64.b64encode(content).decode())


def test_animated_png_is_rejected():
    output = BytesIO()
    Image.new("RGB", (1024, 1024), "red").save(output, format="PNG", save_all=True,
        append_images=[Image.new("RGB", (1024, 1024), "blue")], duration=100, loop=0)
    with pytest.raises(ServiceError):
        validate_reference("data:image/png;base64," + base64.b64encode(output.getvalue()).decode())
