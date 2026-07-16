"""Repository importing: ZIP/TAR/source-file uploads and URL fetching.

Everything lands under data/imports/<name>/ (local filesystem storage) and
is then indexed by the normal pipeline. Supports:

  - uploaded .zip / .tar.gz / .tgz / .tar archives (safely extracted -
    path-traversal entries are rejected)
  - uploaded individual source files (grouped into one folder)
  - URLs: GitHub repo links (https://github.com/owner/repo[/tree/branch])
    are resolved via the zipball API; direct archive URLs are downloaded
"""
import io
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests
from loguru import logger

from app.config import DATA_DIR

IMPORT_DIR = DATA_DIR / "imports"
IMPORT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
_GITHUB_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/tree/([\w./-]+))?/?$")


def _clean(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name).strip("._") or "imported_repo"


def _fresh_dest(name: str) -> Path:
    dest = IMPORT_DIR / _clean(name)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _collapse_single_root(dest: Path) -> None:
    """GitHub zipballs wrap everything in 'repo-sha/'; lift it up."""
    entries = [e for e in dest.iterdir() if not e.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(dest / child.name))
        inner.rmdir()


def _extract_zip(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                logger.warning("skipping unsafe zip entry: {}", member.filename)
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
    _collapse_single_root(dest)


def _extract_tar(data: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dest, filter="data")   # py3.12+: blocks traversal/links
    _collapse_single_root(dest)


def import_files(name: str, files: list[tuple[str, bytes]]) -> str:
    """Import uploaded files. One archive -> extracted repo; otherwise the
    files are grouped as-is into one folder. Returns the repo root path."""
    total = sum(len(d) for _, d in files)
    if total > MAX_ARCHIVE_BYTES:
        raise ValueError(f"upload too large ({total // 1_048_576} MB, "
                         f"limit {MAX_ARCHIVE_BYTES // 1_048_576} MB)")
    first_name = files[0][0]
    dest = _fresh_dest(name or Path(first_name).stem)

    if len(files) == 1 and first_name.lower().endswith(
            (".zip", ".tar.gz", ".tgz", ".tar")):
        fname, data = files[0]
        logger.info("extracting uploaded archive {} ({} kB) -> {}",
                    fname, len(data) // 1024, dest)
        if fname.lower().endswith(".zip"):
            _extract_zip(data, dest)
        else:
            _extract_tar(data, dest)
    else:
        for fname, data in files:
            safe = _clean(Path(fname).name)
            (dest / safe).write_bytes(data)
        logger.info("imported {} loose file(s) -> {}", len(files), dest)
    return str(dest)


def import_url(url: str, name: str = "") -> str:
    """Fetch a repository from a URL. GitHub repo links are resolved to
    zipballs; anything else must be a direct archive URL."""
    url = url.strip()
    gh = _GITHUB_RE.match(url)
    if gh:
        owner, repo, branch = gh.groups()
        fetch = f"https://api.github.com/repos/{owner}/{repo}/zipball" + \
                (f"/{branch}" if branch else "")
        default_name = f"{owner}_{repo}"
        kind = "zip"
    else:
        fetch = url
        default_name = Path(url.split("?")[0]).stem.replace(".tar", "")
        low = url.split("?")[0].lower()
        if low.endswith(".zip"):
            kind = "zip"
        elif low.endswith((".tar.gz", ".tgz", ".tar")):
            kind = "tar"
        else:
            raise ValueError("URL must be a GitHub repository link or a direct "
                             ".zip/.tar.gz archive URL")

    logger.info("fetching {} -> {}", url, fetch)
    with requests.get(fetch, timeout=120, stream=True,
                      headers={"User-Agent": "voice-code-assistant"}) as resp:
        resp.raise_for_status()
        buf, size = io.BytesIO(), 0
        for chunk in resp.iter_content(chunk_size=1 << 16):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ValueError("archive exceeds the 250 MB import limit")
            buf.write(chunk)
    data = buf.getvalue()
    dest = _fresh_dest(name or default_name)
    (_extract_zip if kind == "zip" else _extract_tar)(data, dest)
    logger.info("fetched {} kB -> {}", len(data) // 1024, dest)
    return str(dest)
