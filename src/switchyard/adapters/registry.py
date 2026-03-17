"""Adapter registry for backend adapters."""

from __future__ import annotations

from builtins import list as builtin_list
from collections import defaultdict

from switchyard.adapters.base import BackendAdapter
from switchyard.schemas.backend import (
    BackendHealthState,
    BackendInstance,
    BackendStatusSnapshot,
    DeviceClass,
)


class ServingTargetSnapshot:
    """Resolved backend view for one logical serving target."""

    def __init__(
        self,
        *,
        serving_target: str,
        deployments: list[BackendStatusSnapshot],
    ) -> None:
        self.serving_target = serving_target
        self.deployments = deployments

    @property
    def healthy(self) -> list[BackendStatusSnapshot]:
        return [
            deployment
            for deployment in self.deployments
            if deployment.health.state is not BackendHealthState.UNAVAILABLE
        ]

    @property
    def supports_streaming(self) -> list[BackendStatusSnapshot]:
        return [
            deployment
            for deployment in self.deployments
            if deployment.capabilities.supports_streaming
        ]

    @property
    def warm(self) -> list[BackendStatusSnapshot]:
        return [
            deployment
            for deployment in self.deployments
            if deployment.health.warmed_models
        ]

    @property
    def preferred(self) -> list[BackendStatusSnapshot]:
        return sorted(
            self.deployments,
            key=lambda deployment: (
                deployment.deployment.configured_priority if deployment.deployment else 100,
                -(
                    deployment.deployment.configured_weight
                    if deployment.deployment is not None
                    else 1.0
                ),
                deployment.name,
            ),
        )

    @property
    def instance_inventory(self) -> list[BackendInstance]:
        inventory: list[BackendInstance] = []
        for deployment in self.deployments:
            inventory.extend(deployment.instance_inventory)
        return inventory

    @property
    def healthy_instances(self) -> list[BackendInstance]:
        return [
            instance
            for instance in self.instance_inventory
            if instance.health is not None
            and instance.health.state is not BackendHealthState.UNAVAILABLE
        ]

    @property
    def preferred_instances(self) -> list[BackendInstance]:
        deployment_order = {
            deployment.name: index for index, deployment in enumerate(self.preferred)
        }
        return sorted(
            self.instance_inventory,
            key=lambda instance: (
                deployment_order.get(instance.metadata.get("deployment_name", ""), 10_000),
                0
                if instance.health is not None
                and instance.health.state is not BackendHealthState.UNAVAILABLE
                else 1,
                instance.instance_id,
            ),
        )


class AdapterRegistry:
    """In-memory registry of backend adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BackendAdapter] = {}
        self._serving_targets: dict[str, builtin_list[str]] = defaultdict(list)

    def register(self, adapter: BackendAdapter) -> None:
        """Register an adapter by its unique name."""

        if adapter.name in self._adapters:
            msg = f"adapter '{adapter.name}' is already registered"
            raise ValueError(msg)
        self._adapters[adapter.name] = adapter
        for target in self._infer_serving_targets(adapter):
            if adapter.name not in self._serving_targets[target]:
                self._serving_targets[target].append(adapter.name)

    def get(self, name: str) -> BackendAdapter:
        """Return a registered adapter by name."""

        try:
            return self._adapters[name]
        except KeyError as exc:
            msg = f"adapter '{name}' is not registered"
            raise KeyError(msg) from exc

    def list(self) -> builtin_list[BackendAdapter]:
        """Return adapters in registration order."""

        return builtin_list(self._adapters.values())

    def names(self) -> builtin_list[str]:
        """Return registered adapter names in registration order."""

        return builtin_list(self._adapters)

    def serving_targets(self) -> builtin_list[str]:
        """Return known logical serving targets in registration order."""

        return builtin_list(self._serving_targets)

    def names_for_target(self, serving_target: str) -> builtin_list[str]:
        """Return backend deployment names registered for a logical target."""

        return builtin_list(self._serving_targets.get(serving_target, []))

    def get_for_target(
        self,
        serving_target: str,
        *,
        pinned_backend_name: str | None = None,
    ) -> builtin_list[BackendAdapter]:
        """Resolve candidate adapters for a logical serving target."""

        if pinned_backend_name is not None:
            adapter = self.get(pinned_backend_name)
            if pinned_backend_name not in self._serving_targets.get(serving_target, []):
                msg = (
                    f"adapter '{pinned_backend_name}' is not registered for serving target "
                    f"'{serving_target}'"
                )
                raise KeyError(msg)
            return [adapter]

        names = self._serving_targets.get(serving_target, [])
        return [self._adapters[name] for name in names]

    async def snapshots_for_target(
        self,
        serving_target: str,
        *,
        pinned_backend_name: str | None = None,
    ) -> ServingTargetSnapshot:
        """Return resolved backend snapshots for a logical serving target."""

        deployments: list[BackendStatusSnapshot] = []
        for adapter in self.get_for_target(
            serving_target,
            pinned_backend_name=pinned_backend_name,
        ):
            snapshot = await adapter.status()
            inventory = self._infer_instance_inventory(adapter)
            if inventory:
                if not snapshot.instance_inventory:
                    snapshot.instance_inventory = inventory
                if snapshot.deployment is not None and not snapshot.deployment.instances:
                    snapshot.deployment.instances = list(snapshot.instance_inventory)
            deployments.append(snapshot)
        return ServingTargetSnapshot(serving_target=serving_target, deployments=deployments)

    async def instance_inventory_for_target(
        self,
        serving_target: str,
        *,
        pinned_backend_name: str | None = None,
    ) -> builtin_list[BackendInstance]:
        """Return flattened instance inventory for one logical serving target."""

        snapshot = await self.snapshots_for_target(
            serving_target,
            pinned_backend_name=pinned_backend_name,
        )
        return builtin_list(snapshot.instance_inventory)

    def _infer_serving_targets(self, adapter: BackendAdapter) -> builtin_list[str]:
        model_config = getattr(adapter, "model_config", None)
        if model_config is not None:
            serving_target = getattr(model_config, "serving_target", None)
            alias = getattr(model_config, "alias", None)
            if isinstance(serving_target, str) and serving_target:
                return [serving_target]
            if isinstance(alias, str) and alias:
                return [alias]

        capability_metadata = getattr(adapter, "_capability_metadata", None)
        serving_targets = getattr(capability_metadata, "serving_targets", None)
        if isinstance(serving_targets, list) and serving_targets:
            return [target for target in serving_targets if isinstance(target, str) and target]
        default_model = getattr(capability_metadata, "default_model", None)
        if isinstance(default_model, str) and default_model:
            return [default_model]
        model_ids = getattr(capability_metadata, "model_ids", None)
        if isinstance(model_ids, list) and model_ids:
            first = model_ids[0]
            if isinstance(first, str) and first:
                return [first]
        return []

    def _infer_instance_inventory(
        self,
        adapter: BackendAdapter,
    ) -> builtin_list[BackendInstance]:
        model_config = getattr(adapter, "model_config", None)
        instances = getattr(model_config, "instances", None)
        capability_metadata = getattr(adapter, "_capability_metadata", None)
        if not instances:
            return []

        default_device_class = getattr(
            capability_metadata,
            "device_class",
            DeviceClass.REMOTE,
        )
        model_identifier = getattr(model_config, "model_identifier", None)
        backend_type = getattr(adapter, "backend_type", None)
        inventory: builtin_list[BackendInstance] = []
        for instance in instances:
            to_backend_instance = getattr(instance, "to_backend_instance", None)
            if (
                callable(to_backend_instance)
                and model_identifier is not None
                and backend_type is not None
            ):
                inventory.append(
                    to_backend_instance(
                        backend_type=backend_type,
                        default_device_class=default_device_class,
                        model_identifier=model_identifier,
                    )
                )
                inventory[-1].metadata["deployment_name"] = adapter.name
        return inventory
