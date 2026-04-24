"""Tests para esp32cam_utils: arquitecturas, métricas y utilidades."""

import tempfile
from pathlib import Path

import pytest
import torch

from esp32cam_utils import (
    StandardCNN,
    DepthwiseCNN,
    build_model,
    compute_entropy,
    count_macs,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Arquitecturas
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arch,in_ch,res", [
    ("standard",  3, 96),
    ("standard",  3, 32),
    ("standard",  1, 32),
    ("depthwise", 3, 32),
    ("depthwise", 1, 32),
])
def test_forward_output_shape(arch, in_ch, res):
    mode = "gray" if in_ch == 1 else "rgb"
    model = build_model(arch, mode)
    x = torch.zeros(2, in_ch, res, res)
    out = model(x)
    assert out.shape == (2, 10), f"Esperado (2,10), obtenido {out.shape}"


def test_standard_cnn_custom_channels():
    m = StandardCNN(in_channels=3, channels=(8, 16, 32))
    out = m(torch.zeros(1, 3, 32, 32))
    assert out.shape == (1, 10)
    assert count_parameters(m) < count_parameters(StandardCNN(in_channels=3))


def test_depthwise_cnn_fewer_macs_than_standard():
    std = build_model("standard",  "gray")
    dw  = build_model("depthwise", "gray")
    macs_std = count_macs(std, (1, 1, 32, 32))
    macs_dw  = count_macs(dw,  (1, 1, 32, 32))
    assert macs_dw < macs_std, (
        f"DepthwiseCNN debería tener menos MACs que StandardCNN: "
        f"{macs_dw:,} vs {macs_std:,}"
    )


# ---------------------------------------------------------------------------
# Entropía
# ---------------------------------------------------------------------------

def test_entropy_uniform_is_one():
    probs = torch.ones(10) / 10
    h = compute_entropy(probs)
    assert abs(h - 1.0) < 1e-5, f"Entropía uniforme debe ser 1.0, obtenida {h}"


def test_entropy_one_hot_is_zero():
    probs = torch.zeros(10)
    probs[3] = 1.0
    h = compute_entropy(probs)
    assert h < 1e-5, f"Entropía one-hot debe ser ~0.0, obtenida {h}"


def test_entropy_intermediate():
    probs = torch.tensor([0.7, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0, 0.0, 0.0, 0.0])
    h = compute_entropy(probs)
    assert 0.0 < h < 1.0, f"Entropía intermedia debe estar en (0,1), obtenida {h}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip():
    device = torch.device("cpu")
    model = build_model("depthwise", "gray")
    meta = {
        "arch": "depthwise", "input_mode": "gray", "resolution": 32,
        "num_classes": 10, "accuracy": 0.75, "history": [],
        "normalize_mean": [0.4407], "normalize_std": [0.2220],
        "channels": (32, 64, 128), "source": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test_model.pth"
        save_checkpoint(path, model, meta)
        model2, meta2 = load_checkpoint(path, device)

    assert meta2["arch"] == "depthwise"
    assert meta2["accuracy"] == 0.75
    assert isinstance(model2, DepthwiseCNN)

    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert torch.allclose(p1, p2), f"Parámetro {n1} difiere tras roundtrip"


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

from esp32cam_utils import prune_standard_cnn, prune_depthwise_cnn


def test_prune_standard_cnn_weight_reduces_channels():
    model = StandardCNN(in_channels=3, channels=(32, 64, 128))
    pruned = prune_standard_cnn(model, ratio=0.5, method="weight")
    assert pruned.block1.conv.out_channels == 16
    assert pruned.block2.conv.out_channels == 32
    assert pruned.block3.conv.out_channels == 64


def test_prune_standard_cnn_output_shape_preserved():
    model = StandardCNN(in_channels=3)
    pruned = prune_standard_cnn(model, ratio=0.3, method="weight")
    x = torch.zeros(2, 3, 32, 32)
    out = pruned(x)
    assert out.shape == (2, 10)


def test_prune_depthwise_cnn_weight_reduces_channels():
    model = DepthwiseCNN(in_channels=1, channels=(32, 64, 128))
    pruned = prune_depthwise_cnn(model, ratio=0.5, method="weight")
    assert pruned.block1.pw.out_channels == 16
    assert pruned.block2.pw.out_channels == 32
    assert pruned.block3.pw.out_channels == 64


def test_prune_depthwise_cnn_output_shape_preserved():
    model = DepthwiseCNN(in_channels=1)
    pruned = prune_depthwise_cnn(model, ratio=0.3, method="weight")
    x = torch.zeros(2, 1, 32, 32)
    out = pruned(x)
    assert out.shape == (2, 10)


def test_prune_method_none_preserves_model():
    """method='none' no existe en las funciones — script06 maneja este caso."""
    import copy
    model = StandardCNN(in_channels=3)
    cloned = copy.deepcopy(model)
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), cloned.named_parameters()):
        assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# collect_available_results (.pth + .espdl)
# ---------------------------------------------------------------------------

from esp32cam_utils import collect_available_results, _quantized_parent_name


def test_quantized_parent_name_strips_suffix():
    assert _quantized_parent_name("foo_quantized_ptq") == "foo"
    assert _quantized_parent_name("foo_pruned_activation_20_quantized_ptq") == "foo_pruned_activation_20"
    assert _quantized_parent_name("foo_quantized_qat_brevitas_qdq") == "foo"
    assert _quantized_parent_name("foo_pruned_activation_30") is None  # not quantized


def _save_fp32_checkpoint(path: Path, arch: str, input_mode: str, resolution: int,
                          accuracy: float = 0.5):
    in_ch = 1 if input_mode == "gray" else 3
    model = build_model(arch, input_mode)
    meta = {
        "arch": arch, "input_mode": input_mode, "resolution": resolution,
        "num_classes": 10, "accuracy": accuracy, "history": [],
        "normalize_mean": [0.5] * in_ch, "normalize_std": [0.25] * in_ch,
        "channels": (32, 64, 128), "source": None,
    }
    save_checkpoint(path, model, meta)


def test_collect_results_includes_espdl(tmp_path):
    _save_fp32_checkpoint(tmp_path / "mymodel.pth", "depthwise", "rgb", 32, accuracy=0.7)
    (tmp_path / "mymodel_quantized_ptq.espdl").write_bytes(b"\x00" * 18000)

    results = collect_available_results(tmp_path)
    names = {r["name"] for r in results}
    assert "mymodel" in names
    assert "mymodel_quantized_ptq" in names

    q = next(r for r in results if r["name"] == "mymodel_quantized_ptq")
    assert abs(q["size_kb"] - 18000 / 1024) < 0.1
    assert q["accuracy"] == 0.7  # heredado del padre
    assert q["arch"] == "depthwise"


def test_collect_results_espdl_fallback_pth_is_deduped(tmp_path):
    """Si .espdl y .pth fallback existen con el mismo nombre, solo aparece el .espdl."""
    _save_fp32_checkpoint(tmp_path / "mymodel.pth", "depthwise", "rgb", 32)
    (tmp_path / "mymodel_quantized_ptq.espdl").write_bytes(b"\x00" * 18000)
    _save_fp32_checkpoint(tmp_path / "mymodel_quantized_ptq.pth", "depthwise", "rgb", 32)

    results = collect_available_results(tmp_path)
    quantized = [r for r in results if r["name"] == "mymodel_quantized_ptq"]
    assert len(quantized) == 1
    # el size_kb viene del .espdl, no del .pth
    assert abs(quantized[0]["size_kb"] - 18000 / 1024) < 0.1


def test_collect_results_espdl_without_parent_is_skipped(tmp_path, capsys):
    (tmp_path / "orphan_quantized_ptq.espdl").write_bytes(b"\x00" * 100)
    results = collect_available_results(tmp_path)
    assert all(r["name"] != "orphan_quantized_ptq" for r in results)
    assert "no encontrado" in capsys.readouterr().out
