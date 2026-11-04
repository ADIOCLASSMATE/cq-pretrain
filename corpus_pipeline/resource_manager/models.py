"""Explicit, serializable contracts shared by planners and platform adapters.

GPU counts are integer devices. CPU and memory use the same units as the
adapter's quota catalog (normally cores and GiB); the planner never converts
units or infers a platform, image, owner, mount, or shell command.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _name(value, field):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'{field} must be an explicit nonempty string')


def _integer(value, field, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{field} must be an integer >= {minimum}')


def _resource(value, field, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f'{field} must be a finite {"nonnegative" if zero else "positive"} number')


def _argv(value, field, *, optional=False):
    if value is None and optional:
        return None
    if (not isinstance(value, (tuple, list)) or not value
            or any(not isinstance(part, str) or '\x00' in part for part in value)
            or not value[0]):
        raise ValueError(f'{field} must be a nonempty argv list, not a shell string')
    return tuple(value)


@dataclass(frozen=True)
class WorkloadSpec:
    scope: str
    profile: str
    max_cards: int
    command: tuple[str, ...]
    allowed_cards: tuple[int, ...]
    replica_cards: int | None = None
    migration_policy: str = 'disabled'
    workload_kind: str = 'batch'
    min_ready_replicas: int = 0
    readiness_command: tuple[str, ...] | None = None
    drain_command: tuple[str, ...] | None = None
    completion_command: tuple[str, ...] | None = None
    poll_seconds: float = 10

    def __post_init__(self):
        _name(self.scope, 'scope'); _name(self.profile, 'profile')
        _integer(self.max_cards, 'max_cards')
        object.__setattr__(self, 'command', _argv(self.command, 'command'))
        if not isinstance(self.allowed_cards, (tuple, list)) or not self.allowed_cards:
            raise ValueError('allowed_cards must contain explicit positive device counts')
        for cards in self.allowed_cards:
            _integer(cards, 'allowed_cards entry')
        object.__setattr__(self, 'allowed_cards', tuple(sorted(set(self.allowed_cards), reverse=True)))
        if self.replica_cards is not None:
            _integer(self.replica_cards, 'replica_cards')
            if self.replica_cards not in self.allowed_cards:
                raise ValueError('replica_cards must be present in allowed_cards')
        if self.migration_policy not in ('restartable', 'drain', 'disabled'):
            raise ValueError('migration_policy must be restartable, drain or disabled')
        if self.workload_kind not in ('batch', 'service'):
            raise ValueError('workload_kind must be batch or service')
        if self.workload_kind == 'service' and self.replica_cards is None:
            raise ValueError('service workloads require an explicit fixed replica_cards size')
        _integer(self.min_ready_replicas, 'min_ready_replicas', 0)
        if self.min_ready_replicas and self.replica_cards is None:
            raise ValueError('min_ready_replicas requires a fixed replica_cards contract')
        if self.replica_cards is not None and self.min_ready_replicas * self.replica_cards > self.max_cards:
            raise ValueError('minimum ready replicas exceed the workload budget')
        for field in ('readiness_command', 'drain_command', 'completion_command'):
            object.__setattr__(self, field, _argv(getattr(self, field), field, optional=True))
        if self.migration_policy == 'drain' and self.drain_command is None:
            raise ValueError('drain migration requires an explicit drain_command')
        _resource(self.poll_seconds, 'poll_seconds')

    @property
    def effective_cards(self):
        return (self.replica_cards,) if self.replica_cards is not None else self.allowed_cards


@dataclass(frozen=True)
class NodeCapacity:
    """Live FREE resources, not total installed resources.

    Reservations with capacity_accounted=True are already represented in these
    free figures. Every other non-retired reservation is subtracted locally.
    """
    name: str
    gpu: int
    cpu: float
    memory: float
    eligible: bool = True

    def __post_init__(self):
        _name(self.name, 'node name'); _integer(self.gpu, 'node free gpu', 0)
        _resource(self.cpu, 'node free cpu', zero=True)
        _resource(self.memory, 'node free memory', zero=True)
        if type(self.eligible) is not bool:
            raise ValueError('node eligible must be boolean')


@dataclass(frozen=True)
class Quota:
    cards: int
    cpu: float
    memory: float

    def __post_init__(self):
        _integer(self.cards, 'quota cards')
        _resource(self.cpu, 'quota cpu'); _resource(self.memory, 'quota memory')


@dataclass(frozen=True)
class Reservation:
    id: str
    node: str
    cards: int
    cpu: float
    memory: float
    scope: str
    profile: str
    state: str = 'UNKNOWN'
    retired_verified: bool = False
    capacity_accounted: bool = False
    ready: bool = False
    operation_id: str | None = None
    resource_id: str | None = None

    def __post_init__(self):
        for field in ('id', 'node', 'scope', 'profile', 'state'):
            _name(getattr(self, field), field)
        _integer(self.cards, 'reservation cards')
        _resource(self.cpu, 'reservation cpu'); _resource(self.memory, 'reservation memory')
        object.__setattr__(self, 'state', self.state.upper())
        for field in ('retired_verified', 'capacity_accounted', 'ready'):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f'{field} must be boolean')
        for field in ('operation_id', 'resource_id'):
            if getattr(self, field) is not None:
                _name(getattr(self, field), field)

    @property
    def is_retired(self):
        # A formerly retired resource observed active again still reserves its
        # full quota. Absence from the live inventory is never retirement.
        return self.retired_verified and self.state == 'STOPPED'


@dataclass(frozen=True)
class Placement:
    node: str
    quota: Quota

    def __post_init__(self):
        _name(self.node, 'placement node')
        if not isinstance(self.quota, Quota):
            raise ValueError('placement quota must be a Quota')

    @property
    def cards(self):
        return self.quota.cards


@dataclass(frozen=True)
class AllocationPlan:
    placements: tuple[Placement, ...]
    reserved_cards: int
    remaining_cards: int
    reason: str

    @property
    def added_cards(self):
        return sum(placement.quota.cards for placement in self.placements)


@dataclass(frozen=True)
class MigrationPlan:
    source_ids: tuple[str, ...]
    source_nodes: tuple[str, ...]
    placements: tuple[Placement, ...]
    moved_cards: int
    nodes_before: int
    nodes_after: int
