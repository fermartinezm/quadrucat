"""scripts/compare_sims.py on a fake run (no simulator needed)."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_sims", ROOT / "scripts" / "compare_sims.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def _run(tmp, seed, falls, gz=None):
    run = tmp / "runs" / f"go1_catwalk_v3_s{seed}_x"
    (run / "eval").mkdir(parents=True)
    (run / "eval_cat").mkdir()
    (run / "config.yaml").write_text(f"experiment_name: go1_catwalk_v3\nseed: {seed}\n")
    (run / "eval" / "pushes.json").write_text(json.dumps({"mean": {"fall_rate": falls, "lin_vel_rmse": 0.1}}))
    (run / "eval_cat" / "default.json").write_text(json.dumps({"mean": {"register_err_median": 0.01}}))
    if gz is not None:
        (run / "eval" / "gazebo.json").write_text(json.dumps({"mean": {"fall_rate": gz, "lin_vel_rmse": 0.2}}))
    return run


def test_collect_and_report(tmp_path):
    for seed, falls, gz in ((0, 0.0, 0.5), (1, 0.2, None)):
        cs.collect(_run(tmp_path, seed, falls, gz), tmp_path)
    v = tmp_path / "versions" / "v3"
    assert (v / "gazebo" / "s0" / "gazebo.json").exists() and (v / "mujoco" / "s1" / "eval_pushes.json").exists()
    text = cs.build_report("v3", tmp_path)
    assert "| falls | – | 10 ± 14% | 50% |" in text      # MJX (no fall data) | MJX + pushes, 2 seeds | Gazebo, 1 seed
    assert "MJX + pushes (n=2)" in text and "Gazebo (n=1)" in text
    cs.build_summary(tmp_path)
    assert "[v3](v3/report.md)" in (tmp_path / "versions" / "summary.md").read_text()
