import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import json

path = Path(__file__).with_name('baseline_deadline_supervisor.py')
spec = importlib.util.spec_from_file_location('deadline_ops', path)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def test_existing_runner_and_checkpoint_remain_bound_to_predecessor():
    module = SimpleNamespace(MODULE='frozen.supervisor')
    cfg = {'control': '/run/new', 'ops_path': '/run/new/ops.json', 'operational_handoff': {
        'predecessor_ops': '/run/ops.json', 'launcher': '/run/new/launcher.py',
        'inherited_attempts': {'screening_workflow': 1, 'review_start': 1}}}
    ops.install_routes(module, cfg, {'control': '/run/old'})
    assert module.command_area(cfg, 'screening_workflow', 1) == Path('/run/old/commands/screening_workflow/1')
    assert module.runner_command(cfg, 'screening_workflow', 1) == [sys.executable, '-m', 'frozen.supervisor',
        'execute', '--ops', '/run/ops.json', '--name', 'screening_workflow', '--attempt', '1']
    assert module.command_area(cfg, 'review_start', 2) == Path('/run/new/commands/review_start/2')
    assert module.runner_command(cfg, 'review_start', 2)[1] == '/run/new/launcher.py'


def policy():
    return {'deadline_utc': '1970-01-01T08:00:00Z', 'reserve_after_review_seconds': 10800,
            'fallback_assets_per_replica_second': .75, 'throughput_margin': 1.4,
            'minimum_replicas': 32, 'max_replicas': 64}


def test_expected_remaining_load_fits_initial_target():
    result, _ = ops.estimate_policy(policy(), 182000, 0, 32, [], 0)
    assert result['target_replicas'] == 32
    assert not result['capacity_risk']
    assert result['estimated_review_seconds_at_current_ready'] < 8000


def test_measured_stall_requests_ceiling_and_reports_risk():
    result, _ = ops.estimate_policy(policy(), 182000, 100, 32,
                                   [{'time': 0, 'completed': 100, 'ready': 32}], 600)
    assert result['target_replicas'] == 64
    assert result['capacity_risk']
    assert result['measured_assets_per_second'] == 0


def test_deadline_pressure_increases_target_without_exceeding_cap():
    result, _ = ops.estimate_policy(policy(), 182000, 0, 32, [], 14000)
    assert result['target_replicas'] == 64
    assert result['required_replicas'] > 64


def test_terminal_api_failures_are_not_successful_review_throughput():
    valid, blocked = ops.review_completion({'counts': {'state:done': 100,
        'reason:visual_annotation_only_requires_confirmation': 90,
        'reason:api_review_failure_exhausted': 10}})
    assert (valid, blocked) == (90, 10)


def test_second_handoff_keeps_original_workflow_and_first_successor_review_runner(tmp_path):
    original_path = tmp_path / 'original.json'
    original_path.write_text(json.dumps({'control': '/run/original'}))
    first = {'control': '/run/first', 'operational_handoff': {
        'predecessor_ops': str(original_path), 'predecessor_ops_sha256': ops.sha(original_path),
        'launcher': '/run/first/launch.py', 'inherited_attempts': {'screening_workflow': 1, 'review_start': 1}}}
    second = {'control': '/run/second', 'ops_path': '/run/second/ops.json', 'operational_handoff': {
        'predecessor_ops': '/run/first/ops.json', 'launcher': '/run/second/launch.py',
        'inherited_attempts': {'screening_workflow': 1, 'review_start': 2}}}
    module = SimpleNamespace(MODULE='frozen.supervisor')
    ops.install_routes(module, second, first)
    assert module.runner_command(second, 'screening_workflow', 1)[1:3] == ['-m', 'frozen.supervisor']
    assert module.command_area(second, 'screening_workflow', 1) == Path('/run/original/commands/screening_workflow/1')
    assert module.runner_command(second, 'review_start', 2)[1] == '/run/first/launch.py'
    assert module.command_area(second, 'review_start', 2) == Path('/run/first/commands/review_start/2')
    assert module.runner_command(second, 'review_start', 3)[1] == '/run/second/launch.py'
