"""Dashboard file handling; uploads and results stay scoped to each session."""
from pathlib import Path
import hashlib
import io
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
CATEGORIES = ("images", "ghi", "power", "openmeteo", "models")


def safe_name(name):
    name = Path(str(name).replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if not name or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        raise ValueError("Please rename the uploaded file to a valid filename.")
    return name


def stage_inputs(session, uploads, use_existing=False, root=ROOT):
    if not re.fullmatch(r"[a-f0-9]{32}", session):
        raise ValueError("Invalid session identifier")
    root = Path(root).resolve()
    sources = {kind: [] for kind in CATEGORIES}
    digest = hashlib.sha256()
    for kind in CATEGORIES:
        folder = root / "inputs" / kind
        folder.mkdir(parents=True, exist_ok=True)
        if use_existing:
            for p in sorted(folder.iterdir()):
                if p.is_file() and not p.name.startswith((".", "~$")):
                    sources[kind].append((p.name, p))
                    digest.update(f"{kind}/{p.name}/{p.stat().st_mtime_ns}/{p.stat().st_size}".encode())
        else:
            for upload in uploads.get(kind, []):
                name = safe_name(upload.name)
                data = upload.getvalue()
                sources[kind].append((name, data))
                digest.update(kind.encode() + name.encode() + data)
    version = digest.hexdigest()[:20]
    dirs = {}
    inventory = []
    for kind in CATEGORIES:
        dest = root / "inputs" / kind / "_dashboard" / session / version
        dest.mkdir(parents=True, exist_ok=True)
        dirs[kind] = dest
        names = set()
        for name, source in sources[kind]:
            if name.casefold() in names:
                raise ValueError(f"Duplicate filename {name}; rename one before uploading.")
            names.add(name.casefold())
            path = dest / name
            if not path.exists():
                if isinstance(source, Path):
                    shutil.copy2(source, path)
                else:
                    path.write_bytes(source)
            inventory.append({"Type": kind, "File": name, "Size (KB)": round(path.stat().st_size / 1024, 1)})
    return dirs, inventory, version


def archive_outputs(paths, output_dir):
    output_dir = Path(output_dir).resolve()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    return buffer.getvalue()


def clear_session_outputs(session, root=ROOT):
    if not re.fullmatch(r"[a-f0-9]{32}", session):
        raise ValueError("Invalid session identifier")
    base = (Path(root) / "outputs" / "dashboard").resolve()
    target = (base / session).resolve()
    if target.parent != base or target.is_symlink():
        raise ValueError("Output directory is outside this dashboard session")
    if target.exists():
        shutil.rmtree(target)

