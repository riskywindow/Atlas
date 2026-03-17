from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_blocked_imports(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_control_plane_modules_import_without_optional_worker_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockOptionalWorkers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlx_lm" or fullname.startswith("mlx_lm."):
                    raise ModuleNotFoundError(fullname)
                if fullname == "vllm" or fullname.startswith("vllm."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalWorkers())

        import switchyard.adapters
        import switchyard.config
        import switchyard.gateway.app
        import switchyard.runtime
        import switchyard.schemas.backend

        assert switchyard.adapters.AdapterRegistry.__name__ == "AdapterRegistry"
        assert switchyard.runtime.RuntimeGenerationResult.__name__ == "RuntimeGenerationResult"
        """
    )

    result = _run_blocked_imports(script)

    assert result.returncode == 0, result.stderr


def test_optional_worker_symbols_resolve_without_importing_third_party_packages() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockOptionalWorkers(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mlx_lm" or fullname.startswith("mlx_lm."):
                    raise ModuleNotFoundError(fullname)
                if fullname == "vllm" or fullname.startswith("vllm."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalWorkers())

        from switchyard.adapters import MLXLMAdapter, VLLMMetalAdapter
        from switchyard.runtime import MLXLMChatRuntime, VLLMMetalChatRuntime

        assert MLXLMAdapter.__name__ == "MLXLMAdapter"
        assert VLLMMetalAdapter.__name__ == "VLLMMetalAdapter"
        assert MLXLMChatRuntime.__name__ == "MLXLMChatRuntime"
        assert VLLMMetalChatRuntime.__name__ == "VLLMMetalChatRuntime"
        """
    )

    result = _run_blocked_imports(script)

    assert result.returncode == 0, result.stderr
