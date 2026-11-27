"""Train on a Kaggle GPU from the command line: package -> upload -> run -> wait -> download.

  python scripts/kaggle_run.py launch --name catwalk-v2 --config configs/cat_walk_v2.yaml
  python scripts/kaggle_run.py launch --name check --config configs/cat_walk_v2.yaml --quick   # smoke test
  python scripts/kaggle_run.py launch --name x --config configs/cat_walk.yaml --no-wait \
      --set env.overrides.reward_config.scales.direct_register=10.0
  python scripts/kaggle_run.py package        # only build the zip (no Kaggle account needed)
  python scripts/kaggle_run.py status | wait | fetch --name catwalk-v2

`launch` does everything: zips the repo (tracked + untracked, not git-ignored files), uploads it as a
new version of the private dataset, pushes notebooks/quadrucat_kaggle.ipynb as a private kernel
`quadrucat-<name>` with GPU + Internet (equivalent to Save & Run All), polls it, and unpacks
quadrucat_results.zip into runs/ and figures/ (logs: runs/kaggle/<name>/). Kernels with different names
run in parallel (Kaggle allows 2 GPU sessions, 30 GPU hours per week).

Setup: `pip install kaggle` and a Kaggle API token in ~/.kaggle (kaggle.json or access_token).
The token is only read by the official Kaggle client; this script never reads, prints or packages it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "quadrucat_kaggle.ipynb"
STATE_DIR = ROOT / "runs" / "kaggle"          # git-ignored
DEFAULT_DATASET = "quadrucat-code"            # created (private) when no quadrucat dataset exists
SECRET_NAMES = {"kaggle.json", "access_token", ".env", "credentials.json"}
DONE_STATES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


# ---------------- packaging ----------------

def repo_files(root: pathlib.Path = ROOT) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root, check=True,
                         capture_output=True).stdout.decode("utf-8")
    names = sorted({n for n in out.split("\0") if n and (root / n).is_file()})
    return [n for n in names if pathlib.PurePosixPath(n).name not in SECRET_NAMES
            and not n.startswith(("runs/", "figures/"))
            and not (n.startswith(".") and "/" in n)]   # local tool folders (.vscode/ etc.)


def build_package(out_zip: pathlib.Path, root: pathlib.Path = ROOT) -> str:
    """Write quadrucat.zip and return its code version (hash of the packaged content)."""
    files = {n: (root / n).read_bytes() for n in repo_files(root)}
    digest = hashlib.sha256()
    for name, data in files.items():
        if pathlib.PurePosixPath(name).suffix in {".py", ".sh", ".yaml", ".toml", ".txt", ".md", ".ipynb"}:
            data = files[name] = data.replace(b"\r\n", b"\n")   # same version on Windows and Linux
        digest.update(name.encode() + b"\0" + data + b"\0")
    version = digest.hexdigest()[:12]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(f"quadrucat/{name}", data)
        zf.writestr("quadrucat/CODE_VERSION.txt", f"{version} git {git.stdout.strip()}\n")
    return version


def render_notebook(template: dict, params: dict) -> dict:
    """Replace `NAME = ...` lines of the parameter cell. Unknown names raise."""
    nb = json.loads(json.dumps(template))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and "# --- parameters" in src and "# --- end parameters" in src:
            break
    else:
        raise ValueError("no parameter cell in the notebook")
    head, tail = src.split("# --- end parameters", 1)
    for name, value in params.items():
        head, n = re.subn(rf"^{name} = .*$", lambda _: f"{name} = {value!r}", head, flags=re.M)
        if n != 1:
            raise ValueError(f"parameter {name!r} not found in the notebook")
    cell["source"] = (head + "# --- end parameters" + tail).splitlines(keepends=True)
    return nb


# ---------------- Kaggle ----------------

def connect():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Missing Kaggle client: pip install kaggle")
    api = KaggleApi()
    api.authenticate()      # exits with instructions when there is no token
    return api, api.get_config_value(api.CONFIG_NAME_USER)


def find_dataset(api, user: str, wanted: str | None) -> tuple[str, bool]:
    """(owner/slug, exists). Without --dataset: the user's only 'quadrucat*' dataset."""
    mine = [d.ref for d in (api.dataset_list(mine=True, search="quadrucat") or []) if d]
    if wanted:
        ref = wanted if "/" in wanted else f"{user}/{wanted}"
        return ref, ref in mine
    hits = [r for r in mine if r.split("/")[-1].startswith("quadrucat") and r.split("/")[0] == user]
    if len(hits) > 1:
        preferred = f"{user}/{DEFAULT_DATASET}"
        if preferred in hits:
            return preferred, True
        sys.exit(f"Several datasets match: {hits}. Choose one with --dataset.")
    return (hits[0], True) if hits else (f"{user}/{DEFAULT_DATASET}", False)


def dataset_state(api, ref: str) -> dict:
    return json.loads(api.dataset_status(ref, format="json"))


def upload_dataset(api, ref: str, exists: bool, version: str, zip_path: pathlib.Path, notes: str) -> None:
    marker = STATE_DIR / "uploaded.json"
    last = json.loads(marker.read_text()) if marker.exists() else {}
    if exists and last == {"dataset": ref, "version": version}:
        print(f"[kaggle] {ref} already holds code {version}")
        return
    before = dataset_state(api, ref)["current_version_number"] if exists else 0
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        shutil.copy(zip_path, folder / "quadrucat.zip")
        meta = {"id": ref, "title": ref.split("/")[1], "licenses": [{"name": "other"}]}
        (folder / "dataset-metadata.json").write_text(json.dumps(meta))
        if exists:
            resp = api.dataset_create_version(str(folder), notes, quiet=True)
        else:
            resp = api.dataset_create_new(str(folder), public=False, quiet=True)
    if getattr(resp, "error", None):
        sys.exit(f"Dataset upload failed: {resp.error}")
    print(f"[kaggle] uploaded code {version} to {ref}, waiting until Kaggle has processed it")
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            st = dataset_state(api, ref)
            if st["status"] == "ready" and (st["current_version_number"] or 0) > before:
                break
        except Exception as e:  # a brand-new dataset can answer 403/404 for a few seconds
            print(f"[kaggle] dataset not visible yet ({type(e).__name__})")
        time.sleep(10)
    else:
        sys.exit("The dataset version was not ready after 15 minutes.")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"dataset": ref, "version": version}))


def kernel_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", f"quadrucat-{name}".lower()).strip("-")
    if len(slug) > 50:
        sys.exit(f"--name is too long ({slug})")
    return slug


def push_kernel(api, user: str, name: str, dataset: str, nb: dict, accelerator: str):
    slug = kernel_slug(name)
    meta = {"id": f"{user}/{slug}", "title": slug.replace("-", " "), "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [dataset]}
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        (folder / "notebook.ipynb").write_text(json.dumps(nb), encoding="utf-8")
        (folder / "kernel-metadata.json").write_text(json.dumps(meta))
        resp = api.kernels_push(str(folder), acc=accelerator)
    if getattr(resp, "error", None):
        sys.exit(f"Kernel push failed: {resp.error}")
    bad = getattr(resp, "invalid_dataset_sources", None)
    if bad:
        sys.exit(f"Kaggle could not attach the dataset: {bad}")
    print(f"[kaggle] running: {resp.url}")


def kernel_state(api, user: str, name: str) -> tuple[str, str]:
    ref = f"{user}/{kernel_slug(name)}"
    try:
        r = api.kernels_status(ref)
    except ValueError as e:     # how the Kaggle client reports 401 / 403
        raise SystemExit(f"Kaggle refused access to {ref}: the login was revoked or expired "
                         f"(run `python -m kaggle auth login --force`), or the kernel does not exist.") from e
    return r.status.name, r.failure_message or ""


def wait(api, user: str, name: str, every: float = 60.0, timeout_h: float = 12.5) -> str:
    t0, last, errors = time.time(), None, 0
    while time.time() - t0 < timeout_h * 3600:
        try:
            state, msg = kernel_state(api, user, name)
            errors = 0
        except Exception as e:  # network hiccup: keep polling
            errors += 1
            if errors > 10:
                raise
            print(f"[kaggle] status check failed ({type(e).__name__}), retrying")
            time.sleep(every)
            continue
        if state != last:
            print(f"[kaggle] {name}: {state} {msg} ({(time.time() - t0) / 60:.0f} min)", flush=True)
            last = state
        if state in DONE_STATES:
            return state
        time.sleep(every)
    sys.exit(f"{name} did not finish in {timeout_h} h")


def unpack_results(zip_path: pathlib.Path, log_dir: pathlib.Path, root: pathlib.Path = ROOT) -> list[str]:
    """runs/ and figures/ go to the repo, anything else to log_dir. Returns the new run dirs."""
    runs = set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            parts = pathlib.PurePosixPath(info.filename).parts
            if info.is_dir() or ".." in parts or not parts or ":" in parts[0] or info.filename.startswith("/"):
                continue
            dest = root if parts[0] in ("runs", "figures") and len(parts) > 2 else log_dir
            zf.extract(info, dest)
            if parts[0] == "runs" and len(parts) > 2 and not parts[1].startswith(("smoke", "passive")):
                runs.add(f"runs/{parts[1]}")
    return sorted(runs)


def fetch(api, user: str, name: str) -> list[str]:
    log_dir = STATE_DIR / name
    log_dir.mkdir(parents=True, exist_ok=True)
    ref = f"{user}/{kernel_slug(name)}"
    api.kernels_output(ref, str(log_dir), file_pattern=r"quadrucat_results\.zip$", force=True)
    zip_path = log_dir / "quadrucat_results.zip"
    if not zip_path.exists():
        print(f"[kaggle] no quadrucat_results.zip; downloading raw run files. Log: {log_dir}")
        api.kernels_output(ref, str(log_dir), file_pattern=r"(^|/)runs/.*\.(jsonl|json|yaml|pkl|log)$", force=True)
        return []
    runs = unpack_results(zip_path, log_dir)
    failures = log_dir / "failures.txt"
    if failures.exists() and failures.read_text().strip():
        print("[kaggle] WARNING, some commands failed:\n" + failures.read_text())
    print(f"[kaggle] results: {runs or 'no full run (quick mode?)'}; log in {log_dir}")
    if runs:   # versions/<v>/mujoco/ + report.md for runs of a known version
        subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_sims.py"), "collect", *runs], cwd=ROOT,
                       capture_output=True)
    for run in runs:
        for rel in ("eval_cat/default.json", "eval/pushes.json"):
            p = ROOT / run / rel
            if p.exists():
                mean = json.loads(p.read_text()).get("mean", {})
                print(f"  {run} {rel}: " + "  ".join(f"{k}={v:.3f}" for k, v in mean.items()))
    return runs


# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("launch", help="package, upload, run, wait and download")
    p.add_argument("--name", required=True, help="kernel name suffix; reuse it to push a new version")
    p.add_argument("--config", nargs="+", required=True, help="one or more configs, trained in sequence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--notes", default="")
    p.add_argument("--quick", action="store_true", help="tests + smoke training only")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--dataset", help="dataset slug (default: your only quadrucat* dataset, or a new one)")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle machine shape (T4 x2)")
    p.add_argument("--no-wait", action="store_true")
    p = sub.add_parser("package", help="only build the zip")
    p.add_argument("--out", default=str(STATE_DIR / "quadrucat.zip"))
    for c in ("status", "wait", "fetch"):
        p = sub.add_parser(c)
        p.add_argument("--name", required=True)
    args = ap.parse_args()

    if args.cmd == "package":
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"code version {build_package(out)} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    if args.cmd == "launch":
        for c in args.config:
            if pathlib.PurePath(c).is_absolute() or not (ROOT / c).is_file():
                sys.exit(f"--config must be a path inside the repo: {c}")
        for s in args.set:
            if not re.fullmatch(r"[\w.]+=[\w.,+\-\[\]]+", s):
                sys.exit(f"--set value not supported (no spaces or quotes): {s}")
        if re.search(r"[^\w .,;:+\-=()%/]", args.notes):
            sys.exit("--notes: use only letters, numbers, spaces and .,;:+-=()%/")

    api, user = connect()
    if args.cmd == "status":
        print(*kernel_state(api, user, args.name))
    elif args.cmd == "wait":
        state = wait(api, user, args.name)
        fetch(api, user, args.name)
        sys.exit(0 if state == "COMPLETE" else 1)
    elif args.cmd == "fetch":
        fetch(api, user, args.name)
    else:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = STATE_DIR / "quadrucat.zip"
        version = build_package(zip_path)
        dataset, exists = find_dataset(api, user, args.dataset)
        upload_dataset(api, dataset, exists, version, zip_path, f"code {version}")
        params = {"QUICK": args.quick, "CONFIG_FILES": [c.replace("\\", "/") for c in args.config],
                  "EXPECTED_CODE": version, "SEED": args.seed, "NUM_ENVS": args.num_envs,
                  "EXTRA_SET": " ".join(args.set), "NOTES": args.notes}
        nb = render_notebook(json.loads(NOTEBOOK.read_text(encoding="utf-8")), params)
        push_kernel(api, user, args.name, dataset, nb, args.accelerator)
        (STATE_DIR / f"{args.name}.json").write_text(json.dumps(params | {"dataset": dataset}, indent=2))
        if not args.no_wait:
            time.sleep(20)   # a re-pushed kernel can briefly report the previous version's state
            state = wait(api, user, args.name)
            fetch(api, user, args.name)
            sys.exit(0 if state == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
