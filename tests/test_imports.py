# tests/test_imports.py
import subprocess
import sys


def test_config_results_report_never_import_torch_or_datasets():
    code = (
        "import sys, llm_eval.config, llm_eval.evaluation.results, llm_eval.evaluation.report, "
        "llm_eval.cli;"
        "print(sorted(m for m in ('torch', 'transformers', 'datasets') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    assert out.strip() == "[]"
