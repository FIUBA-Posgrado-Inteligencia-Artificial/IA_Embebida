"""Tests para el pipeline de cuantización (scripts 03a/03b/03c)."""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from esp32cam_utils import build_model
from quantize_utils import generate_model_data_h


# ---------------------------------------------------------------------------
# quantize_utils — header generation
# ---------------------------------------------------------------------------

def test_generate_model_data_h_with_existing_espdl():
    """Con un .espdl válido, el header incluye los bytes y los constants correctos."""
    meta = {"input_mode": "gray", "resolution": 32}
    classes = ["cat", "dog", "bird"]
    raw_bytes = b"\xDE\xAD\xBE\xEF\x01\x02"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        espdl = tmp / "model.espdl"
        espdl.write_bytes(raw_bytes)
        firmware = tmp / "firmware"

        header = generate_model_data_h(
            espdl_path=espdl,
            firmware_dir=firmware,
            meta=meta,
            mean_list=[0.5],
            std_list=[0.25],
            classes=classes,
            source_script="test_script.py",
        )
        content = header.read_text()

    assert "MODEL_INPUT_W          32" in content
    assert "MODEL_INPUT_H          32" in content
    assert "MODEL_INPUT_CHANNELS   1" in content
    assert "MODEL_INPUT_MEAN_0     127" in content     # int(0.5*255)
    assert "MODEL_INPUT_STD_0      63" in content      # int(0.25*255)
    assert "MODEL_NUM_CLASSES      3" in content
    assert '"cat", "dog", "bird"' in content
    assert "0xDE, 0xAD, 0xBE, 0xEF" in content
    assert "model_data_len = 6" in content
    assert "test_script.py" in content


def test_generate_model_data_h_aligns_model_data_array():
    """El array model_data[] debe llevar __attribute__((aligned(16))): esp-dl v3
    sólo mapea el modelo in-place desde flash si está alineado a 16 bytes; si
    no, cae a un camino de copia a heap con capability de PSRAM que en placas
    sin PSRAM devuelve NULL y esp-dl no chequea antes de escribir (crash)."""
    meta = {"input_mode": "gray", "resolution": 32}
    classes = ["cat", "dog", "bird"]
    raw_bytes = b"\xDE\xAD\xBE\xEF\x01\x02"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        espdl = tmp / "model.espdl"
        espdl.write_bytes(raw_bytes)
        firmware = tmp / "firmware"

        header = generate_model_data_h(
            espdl_path=espdl,
            firmware_dir=firmware,
            meta=meta,
            mean_list=[0.5],
            std_list=[0.25],
            classes=classes,
            source_script="test_script.py",
        )
        content = header.read_text()

    assert "static const uint8_t model_data[] __attribute__((aligned(16))) = {" in content


def test_generate_model_data_h_without_espdl_falls_back_to_empty_array():
    """Si el .espdl no existe, se genera un header con array vacío — útil para
    entornos donde esp-ppq no está instalado."""
    meta = {"input_mode": "rgb", "resolution": 96}
    classes = [f"c{i}" for i in range(10)]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        missing = tmp / "does_not_exist.espdl"
        firmware = tmp / "firmware"

        header = generate_model_data_h(
            espdl_path=missing,
            firmware_dir=firmware,
            meta=meta,
            mean_list=[0.4, 0.4, 0.4],
            std_list=[0.2, 0.2, 0.2],
            classes=classes,
            source_script="test_script.py",
        )
        content = header.read_text()
    assert "MODEL_INPUT_CHANNELS   3" in content
    assert "MODEL_INPUT_MEAN_1" in content            # RGB lines present
    assert "MODEL_INPUT_MEAN_2" in content
    assert "modelo no disponible en formato .espdl" in content
    assert "model_data_len = 0" in content


def test_generate_model_data_h_rgb_has_three_mean_std_lines():
    """En modo RGB los defines _MEAN_{0,1,2} y _STD_{0,1,2} están presentes."""
    meta = {"input_mode": "rgb", "resolution": 32}
    classes = ["a"]

    with tempfile.TemporaryDirectory() as tmp:
        firmware = Path(tmp) / "firmware"
        header = generate_model_data_h(
            espdl_path=Path(tmp) / "absent.espdl",
            firmware_dir=firmware,
            meta=meta,
            mean_list=[0.1, 0.2, 0.3],
            std_list=[0.05, 0.1, 0.15],
            classes=classes,
            source_script="t.py",
        )
        content = header.read_text()
    for i, mean in enumerate([25, 51, 76]):    # int(m*255)
        assert f"MODEL_INPUT_MEAN_{i}     {mean}" in content
    for i, std in enumerate([12, 25, 38]):
        assert f"MODEL_INPUT_STD_{i}      {std}" in content


# ---------------------------------------------------------------------------
# Helpers para cargar scripts con guiones en el nombre (no se pueden importar
# normalmente porque Python no acepta guiones en módulos).
# ---------------------------------------------------------------------------

def _load_script_module(script_name: str):
    """Carga un script-<con-guiones>.py como módulo."""
    here = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        script_name.replace("-", "_"), here / "scripts" / f"{script_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_training_inputs(in_ch: int, batches: int = 2, batch_size: int = 4):
    """Batches sintéticos para un smoke test: forma correcta, valores aleatorios."""
    from torch.utils.data import DataLoader, TensorDataset
    xs = torch.randn(batches * batch_size, in_ch, 32, 32)
    ys = torch.randint(0, 10, (batches * batch_size,))
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size)


# ---------------------------------------------------------------------------
# 04b — torchao QAT smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_torchao_qat_smoke_preserves_topology():
    """El strip del modelo QAT-PT2E preserva la misma cantidad de Conv2d y
    Linear que el modelo FP32 original."""
    m03b = _load_script_module("script04b-quantize-torchao")

    meta = {"arch": "depthwise", "input_mode": "gray", "resolution": 32,
            "channels": (8, 16, 32), "num_classes": 10}
    device = torch.device("cpu")
    src = build_model("depthwise", "gray", channels=(8, 16, 32))
    n_conv_src = sum(1 for m in src.modules() if isinstance(m, nn.Conv2d))
    n_lin_src = sum(1 for m in src.modules() if isinstance(m, nn.Linear))

    prepared = m03b.prepare_for_qat(src, meta, device)

    # Un paso de training para asegurar que los fake-quant inicializaron observers
    opt = torch.optim.Adam(prepared.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 32, 32)
    y = torch.tensor([0, 1])
    prepared.train()
    loss = nn.functional.cross_entropy(prepared(x), y)
    loss.backward()
    opt.step()

    stripped = m03b.strip_fake_quant(prepared, meta, device)

    # Topología: mismo nº de Conv2d / Linear "limpios"
    n_conv_out = sum(1 for m in stripped.modules() if type(m) is nn.Conv2d)
    n_lin_out = sum(1 for m in stripped.modules() if type(m) is nn.Linear)
    assert n_conv_out == n_conv_src, f"Conv2d: esperado {n_conv_src}, obtenido {n_conv_out}"
    assert n_lin_out == n_lin_src, f"Linear: esperado {n_lin_src}, obtenido {n_lin_out}"

    stripped.eval()
    out = stripped(x)
    assert out.shape == (2, 10)


# ---------------------------------------------------------------------------
# 04c — brevitas QAT smoke test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("scale_mode", ["float", "power_of_two"])
def test_brevitas_twin_roundtrip(scale_mode):
    """Copiar weights FP32 → twin brevitas → back → FP32 fresco es lossless
    (antes de entrenar; los scales de fake-quant se inicializan, pero en eval
    la salida debe matchear al FP32 original cuando los inputs están dentro
    del rango observado)."""
    m03c = _load_script_module("script04c-quantize-brevitas")

    meta = {"arch": "standard", "input_mode": "gray", "num_classes": 10,
            "channels": (8, 16, 32)}
    src = build_model("standard", "gray", channels=(8, 16, 32))
    twin = m03c.build_brevitas_twin(meta, scale_mode=scale_mode)
    m03c.copy_weights_into_twin(src, twin)

    # Roundtrip a fresh FP32 — weights deben coincidir bit-a-bit
    fresh = build_model("standard", "gray", channels=(8, 16, 32))
    m03c.copy_weights_back_to_fp32(twin, fresh)

    for (n_src, p_src), (n_fresh, p_fresh) in zip(
        src.named_parameters(), fresh.named_parameters()
    ):
        assert torch.equal(p_src, p_fresh), (
            f"Weight mismatch en {n_src} (scale_mode={scale_mode})"
        )


@pytest.mark.slow
def test_brevitas_qat_smoke_preconditioned():
    """Flujo completo del Puente A: build twin → 1 paso training → strip → forward."""
    m03c = _load_script_module("script04c-quantize-brevitas")

    meta = {"arch": "depthwise", "input_mode": "gray", "num_classes": 10,
            "channels": (8, 16, 32)}
    src = build_model("depthwise", "gray", channels=(8, 16, 32))
    twin = m03c.build_brevitas_twin(meta, scale_mode="float")
    m03c.copy_weights_into_twin(src, twin)

    opt = torch.optim.Adam(twin.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 32, 32)
    y = torch.tensor([0, 1])
    twin.train()
    nn.functional.cross_entropy(twin(x), y).backward()
    opt.step()

    fresh = build_model("depthwise", "gray", channels=(8, 16, 32))
    m03c.copy_weights_back_to_fp32(twin, fresh)
    fresh.eval()
    out = fresh(x)
    assert out.shape == (2, 10)


@pytest.mark.slow
def test_brevitas_onnx_qdq_export():
    """Puente B: el twin con scales PoT exporta un ONNX con nodos QDQ."""
    import onnx
    m03c = _load_script_module("script04c-quantize-brevitas")
    m03c._patch_brevitas_onnx_compat()
    from brevitas.export import export_onnx_qcdq

    meta = {"arch": "standard", "input_mode": "gray", "num_classes": 10,
            "channels": (8, 16, 32)}
    src = build_model("standard", "gray", channels=(8, 16, 32))
    twin = m03c.build_brevitas_twin(meta, scale_mode="power_of_two")
    m03c.copy_weights_into_twin(src, twin)

    # Un paso de training para inicializar los observers
    opt = torch.optim.Adam(twin.parameters(), lr=1e-3)
    x = torch.randn(2, 1, 32, 32)
    y = torch.tensor([0, 1])
    twin.train()
    nn.functional.cross_entropy(twin(x), y).backward()
    opt.step()

    twin.eval()
    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        example = torch.randn(1, 1, 32, 32)
        export_onnx_qcdq(twin, args=example, export_path=str(onnx_path), dynamo=False)
        model = onnx.load(str(onnx_path))

    ops = {n.op_type for n in model.graph.node}
    assert "QuantizeLinear" in ops
    assert "DequantizeLinear" in ops
    assert "Conv" in ops
    assert "Gemm" in ops


# ---------------------------------------------------------------------------
# quantize_utils — selección de target
# ---------------------------------------------------------------------------

def test_resolve_target_platform_c_is_generic_int8():
    from quantize_utils import resolve_target_platform
    from esp_ppq import TargetPlatform
    assert resolve_target_platform("c") is TargetPlatform.ESPDL_INT8


def test_resolve_target_platform_esp32s3_is_s3_int8():
    from quantize_utils import resolve_target_platform
    from esp_ppq import TargetPlatform
    assert resolve_target_platform("esp32s3") is TargetPlatform.ESPDL_S3_INT8


def test_resolve_target_platform_rejects_unknown():
    from quantize_utils import resolve_target_platform
    with pytest.raises(ValueError, match="esp32s3"):
        resolve_target_platform("esp32c6")


@pytest.mark.parametrize("script", [
    "script04a-quantize.py",
    "script04b-quantize-torchao.py",
    "script04c-quantize-brevitas.py",
])
def test_script04_variants_expose_target_flag(script):
    """Los tres script04* son peers: los tres deben aceptar --target."""
    import subprocess
    path = Path(__file__).resolve().parents[1] / "scripts" / script
    r = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "--target" in r.stdout
    assert "esp32s3" in r.stdout


def test_bridge_preconditioned_propagates_target(monkeypatch):
    """script04c, puente A: --target debe llegar a run_esppq_ptq/export_espdl.

    Regresión del hallazgo de review: el puente preconditioned no puede
    ignorar --target silenciosamente (un .espdl del target equivocado no
    falla — da resultados basura en la placa)."""
    m03c = _load_script_module("script04c-quantize-brevitas")

    captured = {}

    def fake_run_esppq_ptq(model, calib_data, meta, target="c"):
        captured["run_esppq_ptq_target"] = target
        return "quantized_graph_stub", True

    def fake_export_espdl(quantized_model, output_path, meta, target="c"):
        captured["export_espdl_target"] = target
        return True

    monkeypatch.setattr(m03c, "build_model", lambda **kw: nn.Identity())
    monkeypatch.setattr(m03c, "copy_weights_back_to_fp32", lambda twin, fresh: None)
    monkeypatch.setattr(m03c, "evaluate", lambda model, loader, device: 0.5)
    monkeypatch.setattr(m03c, "save_checkpoint", lambda path, model, meta: None)
    monkeypatch.setattr(m03c, "run_esppq_ptq", fake_run_esppq_ptq)
    monkeypatch.setattr(m03c, "export_espdl", fake_export_espdl)

    meta = {"arch": "depthwise", "input_mode": "gray", "num_classes": 10,
            "channels": (8, 16, 32)}
    m03c.bridge_preconditioned(
        twin=None, meta=meta, test_loader=None, calib_data=[],
        source_name="dummy", device=torch.device("cpu"), target="esp32s3",
    )

    assert captured["run_esppq_ptq_target"] == "esp32s3"
    assert captured["export_espdl_target"] == "esp32s3"


def test_bridge_qdq_propagates_target(monkeypatch, tmp_path):
    """script04c, puente B: --target debe llegar a espdl_quantize_onnx.

    Antes del fix, bridge_qdq pasaba target="c" hardcodeado: con
    --target esp32s3 --bridge qdq el script producía silenciosamente un
    .espdl con kernels genéricos en C."""
    m03c = _load_script_module("script04c-quantize-brevitas")

    captured = {}

    def fake_export_onnx_qcdq(model, args, export_path, dynamo=False):
        Path(export_path).touch()

    def fake_espdl_quantize_onnx(**kwargs):
        captured["target"] = kwargs.get("target")

    monkeypatch.setattr(m03c, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(m03c, "_patch_brevitas_onnx_compat", lambda: None)
    import brevitas.export
    monkeypatch.setattr(brevitas.export, "export_onnx_qcdq", fake_export_onnx_qcdq)
    import esp_ppq.api
    monkeypatch.setattr(esp_ppq.api, "espdl_quantize_onnx", fake_espdl_quantize_onnx)

    meta = {"arch": "standard", "input_mode": "gray", "num_classes": 10,
            "channels": (8, 16, 32), "resolution": 32}
    twin = nn.Identity()
    m03c.bridge_qdq(
        twin=twin, meta=meta, calib_data=[], source_name="dummy",
        device=torch.device("cpu"), target="esp32s3",
    )

    assert captured["target"] == "esp32s3"
