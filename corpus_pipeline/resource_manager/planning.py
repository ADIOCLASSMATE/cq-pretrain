"""Pure resource planning: no filesystem, SDK, shell or lifecycle operations.

Snapshots describe live free capacity. Pending/unknown requests subtract their
whole quota unless the adapter explicitly marks capacity_accounted. A planner
never releases a budget merely because a request is absent from a snapshot.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from .models import AllocationPlan, MigrationPlan, NodeCapacity, Placement, Quota, Reservation, WorkloadSpec


def deduplicate_reservations(reservations):
    """Deduplicate IDs and acknowledged operation/resource aliases conservatively.

    Conflicting resource identities fail closed. Conflicting observations keep
    the reservation active and unready; only unanimous STOPPED retirement
    evidence releases it. The first local ID is retained for engine lookup.
    """
    rows = tuple(reservations)
    parents = list(range(len(rows)))
    aliases = {}
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    for index, row in enumerate(rows):
        if not isinstance(row, Reservation):
            raise ValueError('reservations must contain Reservation objects')
        tokens = [('id', row.id)]
        if row.resource_id:
            tokens.append(('id', row.resource_id))
        if row.operation_id:
            tokens.append(('operation', row.operation_id))
        for token in tokens:
            if token in aliases:
                old, new = find(aliases[token]), find(index)
                parents[max(old, new)] = min(old, new)
            else:
                aliases[token] = index
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(find(index), []).append(row)
    result = []
    for group in groups.values():
        first = group[0]
        identity = ('node', 'cards', 'cpu', 'memory', 'scope', 'profile')
        if any(any(getattr(row, key) != getattr(first, key) for key in identity) for row in group[1:]):
            raise ValueError('reservation ID has conflicting scope or resource contract')
        resource_ids = {row.resource_id for row in group if row.resource_id}
        operation_ids = {row.operation_id for row in group if row.operation_id}
        if len(resource_ids) > 1 or len(operation_ids) > 1:
            raise ValueError('reservation aliases identify more than one operation or resource')
        same_state = len({row.state for row in group}) == 1
        result.append(replace(first,
            state=first.state if same_state else 'UNKNOWN',
            retired_verified=all(row.is_retired for row in group),
            capacity_accounted=all(row.capacity_accounted for row in group),
            ready=all(row.ready and row.state == 'RUNNING' for row in group),
            resource_id=next(iter(resource_ids), None),
            operation_id=next(iter(operation_ids), None)))
    return tuple(result)


def _owned(spec, reservations):
    return tuple(row for row in reservations
                 if row.scope == spec.scope and row.profile == spec.profile and not row.is_retired)


def reserved_cards(spec, reservations):
    return sum(row.cards for row in _owned(spec, deduplicate_reservations(reservations)))


def available_nodes(nodes, reservations):
    """Apply only reservations not already represented in live free capacity.

    Reservations from other scopes also consume node space. No lifecycle
    operation is implied by accounting for them, and they never become owned.
    """
    indexed = {}
    for node in nodes:
        if not isinstance(node, NodeCapacity):
            raise ValueError('nodes must contain NodeCapacity objects')
        if node.name in indexed and indexed[node.name] != node:
            raise ValueError('node free capacity is ambiguous')
        indexed[node.name] = node
    for row in deduplicate_reservations(reservations):
        if row.is_retired or row.capacity_accounted or row.node not in indexed:
            continue
        node = indexed[row.node]
        indexed[row.node] = replace(node, gpu=max(0, node.gpu - row.cards),
                                   cpu=max(0, node.cpu - row.cpu), memory=max(0, node.memory - row.memory))
    return tuple(indexed.values())


def _quotas(spec, quotas):
    effective = set(spec.effective_cards)
    values = set()
    for quota in quotas:
        if not isinstance(quota, Quota):
            raise ValueError('quotas must contain Quota objects')
        if quota.cards in effective:
            values.add(quota)
    # A same-size quota requiring no fewer resources is never preferable.
    values = [q for q in values if not any(
        other != q and other.cards == q.cards and other.cpu <= q.cpu and other.memory <= q.memory
        for other in values)]
    return tuple(sorted(values, key=lambda q: (-q.cards, q.cpu, q.memory)))


@lru_cache(maxsize=8192)
def _pack_node(node, options, limit):
    """Bounded three-resource knapsack, retaining nondominated CPU/memory paths.

    Greedily taking a large quota can strand usable cards (for example a
    CPU-heavy 5-card quota versus two light 3-card quotas). The frontier retains
    those alternatives while preferring fewer instances for equal coverage.
    """
    limit = min(limit, node.gpu)
    if not node.eligible or limit <= 0:
        return ()
    paths = {0: [(0, 0, ())]}
    for cards in range(1, limit + 1):
        frontier = []
        for quota in options:
            for cpu, memory, chosen in paths.get(cards - quota.cards, ()):
                next_cpu, next_memory = cpu + quota.cpu, memory + quota.memory
                if next_cpu > node.cpu or next_memory > node.memory:
                    continue
                next_chosen = tuple(sorted((*chosen, quota), key=lambda q: (-q.cards, q.cpu, q.memory)))
                candidate = (next_cpu, next_memory, next_chosen)
                count = len(next_chosen)
                if any(c <= next_cpu and m <= next_memory and len(qs) <= count
                       for c, m, qs in frontier):
                    continue
                frontier = [(c, m, qs) for c, m, qs in frontier
                            if not (next_cpu <= c and next_memory <= m and count <= len(qs))]
                frontier.append(candidate)
        if frontier:
            paths[cards] = frontier
    covered = max(paths)
    best = min(paths[covered], key=lambda path: (len(path[2]), path[0], path[1],
                                               tuple((-q.cards, q.cpu, q.memory) for q in path[2])))
    return best[2]


def _validate_global_remaining(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError('global_remaining_cards must be a nonnegative integer or None')


def _allocate(spec, nodes, options, reservations, budget):
    owned = _owned(spec, reservations)
    owned_by_node = {}
    for row in owned:
        owned_by_node[row.node] = owned_by_node.get(row.node, 0) + row.cards
    remaining = budget
    placements = []
    candidates = list(available_nodes(nodes, reservations))
    while remaining > 0 and candidates:
        fitted = [(node, _pack_node(node, options, remaining)) for node in candidates]
        fitted = [(node, packed) for node, packed in fitted if packed]
        if not fitted:
            break
        # Use already-owned nodes before opening new ones. Among new nodes the
        # largest feasible coverage reduces spreading; capacity is measured in
        # legal quotas with CPU/memory constraints, not raw free GPU count.
        node, packed = min(fitted, key=lambda pair: (
            0 if pair[0].name in owned_by_node else 1,
            -owned_by_node.get(pair[0].name, 0),
            -sum(q.cards for q in pair[1]), len(pair[1]), pair[0].name))
        placements.extend(Placement(node.name, quota) for quota in packed)
        remaining -= sum(q.cards for q in packed)
        candidates = [candidate for candidate in candidates if candidate.name != node.name]
    return tuple(placements)


def plan_allocations(spec: WorkloadSpec, nodes, quotas, reservations, *, global_remaining_cards=None):
    """Plan additions inside the workload and independently supplied global cap.

    global_remaining_cards must already include *all* known active and
    uncertain requests; it is not a raw live-platform free count. No cap is
    inferred from a platform. The engine rechecks both budgets before sending.
    """
    if not isinstance(spec, WorkloadSpec):
        raise ValueError('an explicit WorkloadSpec is required')
    _validate_global_remaining(global_remaining_cards)
    reservations = deduplicate_reservations(reservations)
    used = sum(row.cards for row in _owned(spec, reservations))
    free = max(0, spec.max_cards - used)
    if global_remaining_cards is not None:
        free = min(free, global_remaining_cards)
    options = _quotas(spec, quotas)
    if spec.replica_cards is not None and any(row.cards != spec.replica_cards for row in _owned(spec, reservations)):
        return AllocationPlan((), used, free, 'configuration_mismatch')
    if not free:
        return AllocationPlan((), used, 0, 'over_budget' if used > spec.max_cards else 'budget_full')
    placements = _allocate(spec, tuple(nodes), options, reservations, free)
    left = free - sum(placement.quota.cards for placement in placements)
    return AllocationPlan(placements, used, left,
                          'filled' if left == 0 else 'partial_capacity' if placements else 'no_legal_fit')


def plan_consolidation(spec: WorkloadSpec, nodes, quotas, reservations, *, global_remaining_cards=None):
    """Plan complete source-node moves, permitting multiple target placements.

    The engine must drain when requested, freshly recheck readiness, verify all
    source stops, then recheck budgets/capacity before submitting replacements.
    A bounded deterministic search favors the largest node-count reduction;
    it is a placement heuristic, not a claim of a global scheduling optimum.
    """
    if not isinstance(spec, WorkloadSpec):
        raise ValueError('an explicit WorkloadSpec is required')
    _validate_global_remaining(global_remaining_cards)
    if spec.migration_policy == 'disabled':
        return None
    reservations = deduplicate_reservations(reservations)
    owned = _owned(spec, reservations)
    if sum(row.cards for row in owned) > spec.max_cards:
        return None
    if spec.replica_cards is not None and any(row.cards != spec.replica_cards for row in owned):
        return None
    groups = {}
    for row in owned:
        groups.setdefault(row.node, []).append(row)
    if len(groups) < 2:
        return None
    nodes, options = tuple(nodes), _quotas(spec, quotas)
    free_by_node = {node.name: sum(q.cards for q in _pack_node(node, options, spec.max_cards))
                    for node in available_nodes(nodes, reservations)}
    eligible = sorted((name for name, rows in groups.items()
                       if all(row.state == 'RUNNING' for row in rows)),
                      key=lambda name: (free_by_node.get(name, 0),
                                        sum(row.cards for row in groups[name]), name))
    if not eligible:
        return None
    ready_total = sum(row.ready and row.state == 'RUNNING' for row in owned)
    # Retain several subsets at each moved-card total; identical totals can
    # have different destinations after excluding the complete source nodes.
    subsets = {(): 0}
    candidates = []
    for name in eligible:
        additions = {tuple((*selected, name)): moved + sum(row.cards for row in groups[name])
                     for selected, moved in subsets.items()}
        subsets.update(additions)
        if len(subsets) > 256:
            by_total = {}
            for selected, moved in sorted(subsets.items(), key=lambda pair: (-len(pair[0]), pair[1], pair[0])):
                if len(by_total.setdefault(moved, [])) < 4:
                    by_total[moved].append((selected, moved))
            selected_values = [entry for entries in by_total.values() for entry in entries]
            subsets = dict(sorted(selected_values, key=lambda pair: (-len(pair[0]), pair[1], pair[0]))[:255])
            subsets[()] = 0
        # Each new subset contains this source, so candidates are evaluated
        # once rather than repeatedly during the following source iterations.
        for selected, moved in additions.items():
            source_ids = {row.id for node in selected for row in groups[node]}
            if ready_total - sum(row.ready and row.state == 'RUNNING' for row in owned
                                 if row.id in source_ids) < spec.min_ready_replicas:
                continue
            remaining_rows = tuple(row for row in reservations if row.id not in source_ids)
            destinations = tuple(node for node in nodes if node.name not in selected)
            placements = _allocate(spec, destinations, options, remaining_rows, moved)
            if sum(placement.quota.cards for placement in placements) != moved:
                continue
            after_nodes = (set(groups) - set(selected)) | {placement.node for placement in placements}
            if len(after_nodes) >= len(groups):
                continue
            source_tuple = tuple(row.id for name in selected for row in groups[name])
            proposal = MigrationPlan(source_tuple, tuple(selected), placements, moved,
                                     len(groups), len(after_nodes))
            new_nodes = len({placement.node for placement in placements} - set(groups))
            score = (len(after_nodes), new_nodes, moved, len(placements), tuple(selected),
                     tuple((p.node, -p.quota.cards, p.quota.cpu, p.quota.memory) for p in placements))
            candidates.append((score, proposal))
    return min(candidates, key=lambda row: row[0])[1] if candidates else None
