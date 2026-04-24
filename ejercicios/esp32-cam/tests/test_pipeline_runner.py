"""Tests para pipeline_runner."""

from pathlib import Path

import pytest

from pipeline_runner import load_pipelines, PipelineError


YAML_VALID = """
flows:
  default:
    - stage: train
      params:
        arch: depthwise
        epochs: 30
    - stage: quantize
      source: depthwise_gray32
      variant: a
      params:
        mode: ptq
  quick:
    - stage: build
      params:
        port: auto
        monitor: false
"""


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipelines.yaml"
    p.write_text(content)
    return p


def test_load_pipelines_valid(tmp_path: Path):
    flows = load_pipelines(_write(tmp_path, YAML_VALID))
    assert set(flows.keys()) == {"default", "quick"}
    assert len(flows["default"]) == 2
    assert flows["default"][0]["stage"] == "train"
    assert flows["default"][1]["variant"] == "a"


def test_load_pipelines_missing_file(tmp_path: Path):
    with pytest.raises(PipelineError, match="not found"):
        load_pipelines(tmp_path / "missing.yaml")


def test_load_pipelines_missing_flows_key(tmp_path: Path):
    with pytest.raises(PipelineError, match="missing 'flows'"):
        load_pipelines(_write(tmp_path, "not_flows: {}\n"))


def test_load_pipelines_bad_stage_name(tmp_path: Path):
    yaml = """
flows:
  bad:
    - stage: unknown_stage
      params: {}
"""
    with pytest.raises(PipelineError, match="unknown stage 'unknown_stage'"):
        load_pipelines(_write(tmp_path, yaml))


def test_load_pipelines_quantize_without_source(tmp_path: Path):
    yaml = """
flows:
  bad:
    - stage: quantize
      params: {}
"""
    with pytest.raises(PipelineError, match="'source' is required for stage 'quantize'"):
        load_pipelines(_write(tmp_path, yaml))


def test_load_pipelines_invalid_variant(tmp_path: Path):
    yaml = """
flows:
  bad:
    - stage: quantize
      source: x
      variant: z
"""
    with pytest.raises(PipelineError, match="variant must be one of"):
        load_pipelines(_write(tmp_path, yaml))


def test_load_pipelines_top_level_port_injected_into_build_stages(tmp_path: Path):
    """Un `port` top-level se inyecta en cada stage build sin port propio."""
    yaml = """
port: /dev/ttyUSB7
flows:
  demo:
    - stage: train
      params: { arch: depthwise }
    - stage: build
      params: { monitor: true }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    build_stage = flows["demo"][1]
    assert build_stage["params"]["port"] == "/dev/ttyUSB7"
    # El stage train no lleva port (no es build)
    assert "port" not in flows["demo"][0].get("params", {})


def test_load_pipelines_stage_port_overrides_top_level(tmp_path: Path):
    """Un `port` explícito dentro de un stage gana sobre el top-level."""
    yaml = """
port: /dev/ttyUSB7
flows:
  demo:
    - stage: build
      params: { port: /dev/ttyUSB0, monitor: true }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    assert flows["demo"][0]["params"]["port"] == "/dev/ttyUSB0"


def test_load_pipelines_without_top_level_port_leaves_stages_untouched(tmp_path: Path):
    yaml = """
flows:
  demo:
    - stage: build
      params: { monitor: true }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    assert "port" not in flows["demo"][0]["params"]


from pipeline_runner import build_command


def test_build_command_train_no_source():
    stage = {"stage": "train", "params": {
        "arch": "depthwise", "input_mode": "gray", "resolution": 32, "epochs": 30
    }}
    cmd = build_command(stage)
    assert cmd[0:3] == ["uv", "run", "python"]
    assert cmd[3].endswith("script01-train.py")
    assert "--arch" in cmd and "depthwise" in cmd
    assert "--input-mode" in cmd and "gray" in cmd
    assert "--resolution" in cmd and "32" in cmd
    assert "--epochs" in cmd and "30" in cmd
    # No --source for train
    assert "--source" not in cmd


def test_build_command_prune_with_source():
    stage = {"stage": "prune", "source": "depthwise_gray32",
             "params": {"method": "activation", "ratio": 0.3}}
    cmd = build_command(stage)
    assert cmd[3].endswith("script02-prune.py")
    assert "--source" in cmd
    assert cmd[cmd.index("--source") + 1] == "depthwise_gray32"
    assert "--ratio" in cmd and "0.3" in cmd


def test_build_command_quantize_variant_b():
    stage = {"stage": "quantize", "source": "x", "variant": "b",
             "params": {"qat_epochs": 5}}
    cmd = build_command(stage)
    assert cmd[3].endswith("script03b-quantize-torchao.py")
    assert "--qat-epochs" in cmd and "5" in cmd


def test_build_command_quantize_default_variant_is_a():
    stage = {"stage": "quantize", "source": "x", "params": {"mode": "ptq"}}
    cmd = build_command(stage)
    assert cmd[3].endswith("script03a-quantize.py")


def test_build_command_build_stage():
    stage = {"stage": "build", "params": {"port": "auto", "monitor": True, "preflight": True}}
    cmd = build_command(stage)
    assert cmd[3].endswith("script04-build.py")
    assert "--port" in cmd and "auto" in cmd
    assert "--monitor" in cmd
    assert "--preflight" in cmd
    assert "--source" not in cmd


def test_build_command_build_stage_with_source():
    stage = {"stage": "build", "source": "my_model_quantized_ptq",
             "params": {"port": "auto", "monitor": False}}
    cmd = build_command(stage)
    assert cmd[3].endswith("script04-build.py")
    assert "--source" in cmd
    assert cmd[cmd.index("--source") + 1] == "my_model_quantized_ptq"


def test_build_command_boolean_false_omits_flag():
    stage = {"stage": "build", "params": {"port": "auto", "monitor": False}}
    cmd = build_command(stage)
    assert "--monitor" not in cmd


def test_build_command_snake_case_to_kebab_case():
    stage = {"stage": "train", "params": {"batch_size": 64, "input_mode": "gray"}}
    cmd = build_command(stage)
    assert "--batch-size" in cmd
    assert "--input-mode" in cmd


from pipeline_runner import run_flow


def test_run_flow_dry_run_prints_commands(capsys, tmp_path: Path):
    yaml = """
flows:
  demo:
    - stage: train
      params: { arch: depthwise, input_mode: gray, resolution: 32 }
    - stage: build
      params: { port: auto, monitor: false }
"""
    config = _write(tmp_path, yaml)
    rc = run_flow("demo", load_pipelines(config), dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "[dry-run]" in out
    assert "script01-train.py" in out
    assert "script04-build.py" in out


def test_run_flow_unknown_flow_raises(tmp_path: Path):
    config = _write(tmp_path, "flows:\n  a:\n    - stage: train\n      params: {}\n")
    with pytest.raises(PipelineError, match="flow 'missing' not found"):
        run_flow("missing", load_pipelines(config), dry_run=True)


from unittest.mock import patch


def test_run_flow_live_invokes_subprocess(tmp_path: Path):
    yaml = """
flows:
  demo:
    - stage: train
      params: { arch: depthwise }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    with patch("pipeline_runner._run_subprocess", return_value=0) as mock_run:
        rc = run_flow("demo", flows, dry_run=False)
    assert rc == 0
    assert mock_run.call_count == 1
    args, _ = mock_run.call_args
    cmd = args[0]
    assert cmd[3].endswith("script01-train.py")


def test_run_flow_fail_fast_stops_on_first_failure(capsys, tmp_path: Path):
    yaml = """
flows:
  demo:
    - stage: train
      params: {}
    - stage: build
      params: { port: auto }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    with patch("pipeline_runner._run_subprocess", side_effect=[3, 0]) as mock_run:
        rc = run_flow("demo", flows, dry_run=False)
    assert rc == 3
    assert mock_run.call_count == 1  # Second stage never runs


def test_run_flow_prints_stage_prefix_on_dry_run(capsys, tmp_path: Path):
    yaml = """
flows:
  demo:
    - stage: quantize
      source: x
      params: { mode: ptq }
"""
    flows = load_pipelines(_write(tmp_path, yaml))
    run_flow("demo", flows, dry_run=True)
    out = capsys.readouterr().out
    assert "[quantize]" in out
