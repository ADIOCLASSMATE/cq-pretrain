"""Local TAR range reads and bounded parallel image decoding. No network path."""
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class ImageReader:
    """Lease cached descriptors so eviction cannot race a concurrent pread.

    Payloads live only in the bounded caller queue and the filesystem page
    cache. There is no SQLite admission lock, downloaded copy, or HTTP fallback.
    """
    def __init__(self, source_root, source_sizes, image_size=512,
                 native_resolution=False, max_open=32, max_bytes=256 << 20):
        self.root = Path(source_root).resolve()
        self.sizes = source_sizes
        self.image_size = image_size
        self.native_resolution = native_resolution
        self.max_open, self.max_bytes = max_open, max_bytes
        self.fds = OrderedDict()
        self.lock = threading.Lock()
        self.counters = dict(images=0, local_bytes=0, reads=0)

    def read_bytes(self, locator):
        relative = locator['shard']
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or '://' in relative:
            raise ValueError(f'Expected a local relative TAR path: {relative}')
        offset, length = int(locator['offset']), int(locator['length'])
        if not 0 < length <= self.max_bytes or offset < 0:
            raise ValueError(f'Invalid image range: {relative}:{offset}+{length}')
        expected = self.sizes[relative]
        if offset + length > expected:
            raise ValueError('Image range extends beyond the immutable TAR')
        with self.lock:
            if relative not in self.fds:
                resolved = (self.root / relative).resolve()
                if not resolved.is_relative_to(self.root):
                    raise ValueError('TAR escapes the configured local source root')
                fd = os.open(resolved, os.O_RDONLY)
                if os.fstat(fd).st_size != expected:
                    os.close(fd)
                    raise RuntimeError(f'Source TAR size changed: {relative}')
                self.fds[relative] = fd
            self.fds.move_to_end(relative)
            fd = os.dup(self.fds[relative])
            while len(self.fds) > self.max_open:
                os.close(self.fds.popitem(last=False)[1])
        try:
            body = os.pread(fd, length, offset)
        finally:
            os.close(fd)
        if len(body) != length or hashlib.sha256(body).hexdigest() != locator['sha256']:
            raise RuntimeError(f'Image integrity failure: {relative}:{offset}')
        with self.lock:
            self.counters['local_bytes'] += length
            self.counters['reads'] += 1
        return body

    def read(self, locator):
        body = self.read_bytes(locator)
        with Image.open(io.BytesIO(body)) as image:
            image.seek(0)
            image.load()
            image = ImageOps.exif_transpose(image).convert('RGB')
            if not self.native_resolution:
                image = ImageOps.fit(image, (self.image_size, self.image_size),
                                     Image.Resampling.LANCZOS, centering=(.5, .5))
            pixels = np.asarray(image, dtype=np.uint8).copy()
        with self.lock:
            self.counters['images'] += 1
        return pixels

    def stats(self):
        with self.lock:
            return dict(self.counters, open_tars=len(self.fds))

    def close(self):
        with self.lock:
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()
