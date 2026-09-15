"""Stage 2: get ONE SoccerNet GSR clip without downloading the whole 11 GB valid.zip.

Usage: uv run python src/track/fetch_soccernet.py                         (list clips + sizes)
       uv run python src/track/fetch_soccernet.py SNGS-033 --labels-only  (just the 2-4 MB labels)
       uv run python src/track/fetch_soccernet.py SNGS-033                (labels + 750 frames)
       uv run python src/track/fetch_soccernet.py sequences_info.json     (one top-level file)
Writes data/soccernet/<clip>/Labels-GameState.json and data/soccernet/<clip>/img1/*.jpg.

How: a zip keeps its table of contents at the END of the file, and Hugging Face answers
"Range" requests (send me bytes 1000-2000 only). So we read the table, then fetch only the
bytes of the files we want. Python's zipfile does the rest; it just thinks it has a local file.
"""
import io
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

URL = "https://huggingface.co/datasets/SoccerNet/SN-GSR-2025/resolve/main/valid.zip"
OUT = Path(__file__).resolve().parents[2] / "data" / "soccernet"
CHUNK = 8 * 1024 * 1024  # fetch 8 MB per request, so a 155 MB clip is ~20 requests, not thousands


class RemoteFile(io.RawIOBase):
    """A read-only file whose bytes live on a web server, fetched on demand."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        self.size = int(head.headers["Content-Length"])
        self.buf_start, self.buf = 0, b""  # last chunk fetched

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = [offset, self.pos + offset, self.size + offset][whence]
        return self.pos

    def read(self, n=-1):
        end = self.size if n < 0 else min(self.pos + n, self.size)
        if end <= self.pos:
            return b""
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            stop = min(max(end, self.pos + CHUNK), self.size) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{stop}"})
            self.buf_start, self.buf = self.pos, urllib.request.urlopen(req).read()
        i = self.pos - self.buf_start
        data = self.buf[i : i + (end - self.pos)]
        self.pos += len(data)
        return data


def main():
    z = zipfile.ZipFile(RemoteFile(URL))
    if len(sys.argv) < 2:
        sizes = defaultdict(int)
        for info in z.infolist():
            sizes[info.filename.split("/")[0]] += info.compress_size
        for clip, size in sorted(sizes.items()):
            print(f"{clip}  {size / 1e6:6.1f} MB")
        print(f"{len(sizes)} clips")
        return

    clip, labels_only = sys.argv[1], "--labels-only" in sys.argv
    wanted = [
        i for i in z.infolist()
        if (i.filename == clip or i.filename.startswith(clip + "/")) and not i.is_dir()
        and (not labels_only or i.filename.endswith(".json"))
    ]
    if not wanted:
        sys.exit(f"no clip called {clip} in valid.zip (run without arguments to list them)")
    total = sum(i.compress_size for i in wanted)
    print(f"{clip}: {len(wanted)} files, {total / 1e6:.1f} MB -> {OUT / clip}")
    for k, info in enumerate(sorted(wanted, key=lambda i: i.header_offset)):  # file order = fewer requests
        z.extract(info, OUT)
        if k % 50 == 0:
            print(f"  {k}/{len(wanted)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
