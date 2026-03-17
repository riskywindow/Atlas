"""CLI for serving a single Switchyard worker process."""

from __future__ import annotations

from collections.abc import Callable

import typer
import uvicorn

from switchyard.adapters.base import BackendAdapter
from switchyard.config import LocalModelConfig, Settings
from switchyard.schemas.backend import BackendType, WorkerTransportType
from switchyard.worker.app import create_worker_app

app = typer.Typer(help="Serve a host-native Switchyard worker process.")

WorkerAdapterBuilder = Callable[[LocalModelConfig], BackendAdapter]
ServerRunner = Callable[..., None]


@app.command("serve")
def serve(
    target: str = typer.Argument(
        ...,
        help="Model alias or backend identity, for example 'mlx-chat' or 'mlx-lm:mlx-chat'.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8101, "--port", min=1, max=65535, help="Bind port."),
    warmup_mode: str = typer.Option(
        "config",
        "--warmup-mode",
        help="Startup warmup behavior: config, eager, or lazy.",
    ),
    warmup_model_id: str | None = typer.Option(
        None,
        "--warmup-model-id",
        help="Optional model identifier override for startup warmup.",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Worker log level."),
) -> None:
    """Serve one configured host-native worker."""

    serve_worker_process(
        settings=Settings(),
        target=target,
        host=host,
        port=port,
        warmup_mode=warmup_mode,
        warmup_model_id=warmup_model_id,
        log_level=log_level,
    )


def serve_worker_process(
    *,
    settings: Settings,
    target: str,
    host: str,
    port: int,
    warmup_mode: str,
    warmup_model_id: str | None,
    log_level: str,
    adapter_builder: WorkerAdapterBuilder | None = None,
    server_runner: ServerRunner = uvicorn.run,
) -> None:
    """Resolve one configured backend and serve it as an HTTP worker."""

    model_config = resolve_worker_model_config(settings=settings, target=target)
    if model_config.worker_transport is not WorkerTransportType.IN_PROCESS:
        msg = (
            f"worker target {target!r} is configured for {model_config.worker_transport.value}; "
            "the worker server only wraps host-native in-process adapters"
        )
        raise typer.BadParameter(msg)

    resolved_warmup = _resolve_warmup_mode(model_config=model_config, warmup_mode=warmup_mode)
    builder = adapter_builder or _build_local_worker_adapter
    adapter = builder(model_config)
    worker_name = _worker_identity(model_config)
    app_instance = create_worker_app(
        adapter,
        worker_name=worker_name,
        warmup_on_start=resolved_warmup,
        warmup_model_id=warmup_model_id,
        log_level=log_level,
    )
    server_runner(app_instance, host=host, port=port, log_level=log_level.lower())


def resolve_worker_model_config(*, settings: Settings, target: str) -> LocalModelConfig:
    """Resolve a worker target by alias or canonical backend identity."""

    matches = [
        model_config
        for model_config in settings.local_models
        if target in {model_config.alias, _worker_identity(model_config)}
    ]
    if not matches:
        msg = (
            f"no configured local model matches {target!r}; "
            "use a model alias or backend identity such as 'mlx-lm:chat-model'"
        )
        raise typer.BadParameter(msg)
    if len(matches) > 1:
        msg = (
            f"target {target!r} is ambiguous; choose one of: "
            + ", ".join(sorted(_worker_identity(model_config) for model_config in matches))
        )
        raise typer.BadParameter(msg)
    return matches[0]


def _resolve_warmup_mode(*, model_config: LocalModelConfig, warmup_mode: str) -> bool:
    normalized = warmup_mode.strip().lower()
    if normalized == "config":
        return model_config.warmup.eager
    if normalized == "eager":
        return True
    if normalized == "lazy":
        return False
    msg = "warmup_mode must be one of: config, eager, lazy"
    raise typer.BadParameter(msg)


def _build_local_worker_adapter(model_config: LocalModelConfig) -> BackendAdapter:
    if model_config.backend_type is BackendType.MLX_LM:
        from switchyard.adapters.mlx_lm import MLXLMAdapter

        return MLXLMAdapter(model_config)
    if model_config.backend_type is BackendType.VLLM_METAL:
        from switchyard.adapters.vllm_metal import VLLMMetalAdapter

        return VLLMMetalAdapter(model_config)
    msg = (
        f"worker serving is not implemented for backend_type "
        f"{model_config.backend_type.value!r}"
    )
    raise typer.BadParameter(msg)


def _worker_identity(model_config: LocalModelConfig) -> str:
    prefix = {
        BackendType.MLX_LM: "mlx-lm",
        BackendType.VLLM_METAL: "vllm-metal",
    }.get(model_config.backend_type, model_config.backend_type.value)
    return f"{prefix}:{model_config.alias}"


def main() -> None:
    """Entrypoint for `python -m switchyard.worker.cli`."""

    app()


if __name__ == "__main__":
    main()
