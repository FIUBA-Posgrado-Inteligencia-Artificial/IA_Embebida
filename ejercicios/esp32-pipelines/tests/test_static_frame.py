import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gen_static_frame.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_static_frame", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rgb888_to_rgb565_packing():
    mod = _load_module()
    assert mod.rgb888_to_rgb565(255, 255, 255) == 0xFFFF
    assert mod.rgb888_to_rgb565(0, 0, 0) == 0x0000
    assert mod.rgb888_to_rgb565(255, 0, 0) == 0xF800
    assert mod.rgb888_to_rgb565(0, 255, 0) == 0x07E0
    assert mod.rgb888_to_rgb565(0, 0, 255) == 0x001F
