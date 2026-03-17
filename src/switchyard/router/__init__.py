"""Routing package."""

__all__ = ["NoRouteAvailableError", "RouterService"]


def __getattr__(name: str) -> object:
    if name in {"NoRouteAvailableError", "RouterService"}:
        from switchyard.router.service import NoRouteAvailableError, RouterService

        return {
            "NoRouteAvailableError": NoRouteAvailableError,
            "RouterService": RouterService,
        }[name]
    msg = f"module 'switchyard.router' has no attribute {name!r}"
    raise AttributeError(msg)
