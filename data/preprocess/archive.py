"""Read one original video at a time from ordinary or concatenated ZIP parts."""
import bisect
import hashlib
import io
from pathlib import Path


class SplitFile(io.RawIOBase):
    """Seekable concatenation; adapted from data/analysis/mp4_metadata.py."""

    def __init__(self, paths):
        self.paths = [Path(p) for p in paths]
        self.ends = [0]
        for path in self.paths:
            self.ends.append(self.ends[-1] + path.stat().st_size)
        self.handles = {}
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid whence")
        position = (0, self.position, self.ends[-1])[whence] + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.ends[-1] - self.position
        blocks = []
        while size > 0 and self.position < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.position) - 1
            if index not in self.handles:
                self.handles[index] = self.paths[index].open("rb")
            handle = self.handles[index]
            handle.seek(self.position - self.ends[index])
            block = handle.read(min(size, self.ends[index + 1] - self.position))
            if not block:
                raise EOFError("Truncated ZIP part")
            blocks.append(block)
            self.position += len(block)
            size -= len(block)
        return b"".join(blocks)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        super().close()


def extract_video(archive, record, destination, expected_crc32):
    """ZipExtFile verifies the member CRC at EOF; also hash original MP4 bytes."""
    source = record["source"]
    info = archive.getinfo(source["member"])
    if info.file_size != source["bytes"] or info.CRC != expected_crc32:
        raise ValueError("ZIP member identity differs from frozen inventory")
    digest = hashlib.sha256()
    with archive.open(info) as src, Path(destination).open("wb") as dst:
        while block := src.read(4 * 1024 * 1024):
            dst.write(block)
            digest.update(block)
    return digest.hexdigest()
