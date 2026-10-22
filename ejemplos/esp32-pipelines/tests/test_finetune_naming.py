import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "script03-finetune.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("script03_finetune", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_output_name_appends_ft_epochs():
    mod = _load_module()
    assert mod.output_name_for("depthwise_rgb32_pruned_activation_30", 5) == \
        "depthwise_rgb32_pruned_activation_30_ft5"


def test_output_name_is_repeatable():
    mod = _load_module()
    once = mod.output_name_for("m_pruned_activation_30", 5)
    assert mod.output_name_for(once + "_pruned_activation_20", 3) == \
        "m_pruned_activation_30_ft5_pruned_activation_20_ft3"
