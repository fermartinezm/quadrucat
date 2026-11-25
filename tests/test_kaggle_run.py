"""Offline parts of scripts/kaggle_run.py (no Kaggle account needed)."""
import importlib.util
import json
import pathlib
import shutil
import subprocess
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts" / "kaggle_run.py")
kr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kr)


def test_render_notebook_sets_parameters_and_cells_compile():
    tpl = json.loads(kr.NOTEBOOK.read_text(encoding="utf-8"))
    nb = kr.render_notebook(tpl, {"QUICK": True, "CONFIG_FILES": ["configs/cat_walk_v2.yaml"], "SEED": 3,
                                  "EXPECTED_CODE": "abc", "NUM_ENVS": None, "EXTRA_SET": "a.b=1", "NOTES": "x"})
    cell = next("".join(c["source"]) for c in nb["cells"] if "# --- parameters" in "".join(c["source"]))
    for line in ("QUICK = True", "CONFIG_FILES = ['configs/cat_walk_v2.yaml']", "SEED = 3", "EXPECTED_CODE = 'abc'",
                 "NUM_ENVS = None", "EXTRA_SET = 'a.b=1'"):
        assert line in cell.splitlines()
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("\n".join("pass" if l.startswith("!") else l for l in "".join(c["source"]).split("\n")), "c", "exec")
    with pytest.raises(ValueError):
        kr.render_notebook(tpl, {"NOT_A_PARAMETER": 1})


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_package_excludes_ignored_and_secret_files(tmp_path):
    repo = tmp_path / "r"
    (repo / "runs").mkdir(parents=True)
    for name, text in {"a.py": "x = 1\r\n", ".gitignore": "runs/\n", "kaggle.json": "{}", "runs/big": "0"}.items():
        (repo / name).write_bytes(text.encode())
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    v1 = kr.build_package(tmp_path / "1.zip", repo)
    names = set(zipfile.ZipFile(tmp_path / "1.zip").namelist())
    assert names == {"quadrucat/.gitignore", "quadrucat/a.py", "quadrucat/CODE_VERSION.txt"}
    assert zipfile.ZipFile(tmp_path / "1.zip").read("quadrucat/CODE_VERSION.txt").decode().split()[0] == v1
    (repo / "a.py").write_bytes(b"x = 1\n")       # line endings do not change the version
    assert kr.build_package(tmp_path / "2.zip", repo) == v1
    (repo / "a.py").write_bytes(b"x = 2\n")
    assert kr.build_package(tmp_path / "3.zip", repo) != v1


def test_unpack_results(tmp_path):
    z = tmp_path / "res.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for n in ("runs/go1_s0/params.pkl", "runs/smoke_s0/config.yaml", "figures/go1_s0/a.png", "train.log",
                  "../evil.txt"):
            zf.writestr(n, "x")
    runs = kr.unpack_results(z, tmp_path / "log", tmp_path / "root")
    assert runs == ["runs/go1_s0"]
    assert (tmp_path / "root/figures/go1_s0/a.png").exists() and (tmp_path / "log/train.log").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_kernel_slug():
    assert kr.kernel_slug("catwalk_v2") == "quadrucat-catwalk-v2"
