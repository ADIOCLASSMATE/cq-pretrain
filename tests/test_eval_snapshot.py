"""CPU/Gloo export roundtrips, EMA restoration and interrupted-write recovery."""
import copy
from contextlib import redirect_stderr
from datetime import timedelta
import io
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

import utils.checkpoint as checkpoint
import utils.eval_snapshot as eval_snapshot
from training.ema import RendererEMA


def _model(tp_rank):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    model.renderer = torch.nn.Linear(2, 2, bias=False).to(torch.bfloat16)
    with torch.no_grad():
        model.backbone.weight.fill_(2+tp_rank)
        model.renderer.weight.copy_(torch.arange(4).reshape(2, 2)+10+tp_rank)
    return model


def _worker(root, rank):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group('gloo', init_method='file://'+str(root/'gloo'),
                            rank=rank, world_size=4, timeout=timedelta(seconds=30))
    groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    tp_rank, dp_rank = rank % 2, rank // 2
    group = groups[tp_rank]
    parallel = SimpleNamespace(get_tensor_model_parallel_rank=lambda:tp_rank,
                               get_data_parallel_group=lambda:group)
    model = _model(tp_rank)
    parameter = model.renderer.weight
    start, end = 2*dp_rank, 2*(dp_rank+1)
    master = parameter.detach().flatten()[start:end].float().clone()
    optimizer = SimpleNamespace(model_float16_groups=[[parameter]],
        shard_fp32_from_float16_groups=[[master]], model_fp32_groups=[], shard_fp32_groups=[],
        _get_model_param_range_map=lambda p:{'param':SimpleNamespace(start=start, end=end)})

    def publish(**kwargs):
        part = parameter.detach().flatten()[start:end].contiguous()
        gathered = [torch.empty_like(part), torch.empty_like(part)]
        dist.all_gather(gathered, part, group=group)
        with torch.no_grad():
            parameter.copy_(torch.cat(gathered).reshape_as(parameter))

    bucket = SimpleNamespace(params=[parameter], start_param_sync=publish)
    handles = {'active':object()}
    wrapped = SimpleNamespace(bucket_groups=[bucket], expert_parallel_bucket_groups=[],
        use_forward_hook=True, remove_forward_pre_hook_handles=handles,
        start_param_sync=publish,
        disable_forward_pre_hook=lambda **kwargs:handles.clear(),
        enable_forward_pre_hook=lambda:handles.update(active=object()))
    cfg = SimpleNamespace(to_dict=lambda:dict(width=2))
    topology = dict(tp=2, dp=2, pp=1, cp=1, optimizer_instances=1)

    with patch.object(eval_snapshot.P, 'state', return_value=parallel), \
         patch.object(eval_snapshot.P, 'tp_size', return_value=2), \
         patch.object(eval_snapshot.P, 'dp_rank', return_value=dp_rank):
        ema = RendererEMA(model, optimizer, wrapped, .9)
        for shadow in ema.shadows:
            shadow.fill_(.25+tp_rank)
        ema.num_updates = 1000
        original = copy.deepcopy(model.state_dict())
        master_before = master.clone()

        def assert_restored():
            assert handles and not ema.active
            torch.testing.assert_close(master, master_before, rtol=0, atol=0)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)

        snapshot = root/'eval_weights'/'step_0001000'
        event = eval_snapshot.save(snapshot, model, wrapped, 1000, cfg, topology, ema,
                                    provenance={'source_revision':'test'})
        assert not event['skipped']
        assert_restored()
        manifest = eval_snapshot.inspect_snapshot(snapshot)
        assert set(manifest['files']) == {'model_tp_000.pt', 'model_tp_001.pt'}
        assert set(p.name for p in snapshot.iterdir()) == set(manifest['files']) | {'manifest.json'}
        assert not {'optimizer', 'data_state', 'rng', 'training_state'} & manifest.keys()
        assert manifest['weight_sources']['renderer'] == 'ema'
        saved = torch.load(snapshot/f'model_tp_{tp_rank:03d}.pt', weights_only=True)
        assert set(saved) == set(original)
        torch.testing.assert_close(saved['backbone.weight'], original['backbone.weight'], rtol=0, atol=0)
        torch.testing.assert_close(saved['renderer.weight'],
                                   torch.full_like(parameter, .25+tp_rank), rtol=0, atol=0)

        restored_model = _model(tp_rank)
        loaded = checkpoint.initialize_model(snapshot, restored_model, cfg)
        assert loaded['source_step'] == 1000
        for name, tensor in restored_model.state_dict().items():
            torch.testing.assert_close(tensor, saved[name], rtol=0, atol=0)

        timestamps = {p.name:p.stat().st_mtime_ns for p in snapshot.iterdir()}
        with patch.object(eval_snapshot, '_save_file', side_effect=AssertionError('must not rewrite')):
            again = eval_snapshot.save(snapshot, model, wrapped, 1000, cfg, topology, ema,
                                        provenance={'attempt':'different'})
        assert again['skipped']
        assert timestamps == {p.name:p.stat().st_mtime_ns for p in snapshot.iterdir()}
        assert_restored()

        online = root/'eval_weights'/'step_0002000'
        eval_snapshot.save(online, model, wrapped, 2000, cfg, topology)
        online_state = torch.load(online/f'model_tp_{tp_rank:03d}.pt', weights_only=True)
        for name, tensor in online_state.items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
        assert eval_snapshot.inspect_snapshot(online)['weight_sources']['renderer'] == 'online'

        incomplete = root/'eval_weights'/'step_0003000'
        real_write = eval_snapshot._save_file

        def fail_one_writer(value, path):
            if rank == 1:
                raise OSError('injected shard disk failure')
            return real_write(value, path)

        with patch.object(eval_snapshot, '_save_file', side_effect=fail_one_writer):
            try:
                eval_snapshot.save(incomplete, model, wrapped, 3000, cfg, topology, ema)
            except RuntimeError as error:
                assert 'injected shard disk failure' in str(error)
            else:
                raise AssertionError('Rank-local I/O failure was not propagated')
        assert not (incomplete/'manifest.json').exists()
        assert_restored()
        eval_snapshot.save(incomplete, model, wrapped, 3000, cfg, topology, ema)
        assert_restored()

        final = root/'eval_weights'/'step_0004000'
        with patch.object(eval_snapshot, 'durable_json', side_effect=OSError('injected commit failure')):
            try:
                eval_snapshot.save(final, model, wrapped, 4000, cfg, topology, ema)
            except RuntimeError as error:
                assert 'injected commit failure' in str(error)
            else:
                raise AssertionError('Commit failure was not propagated')
        assert not (final/'manifest.json').exists()
        assert_restored()
        eval_snapshot.save(final, model, wrapped, 4000, cfg, topology, ema)
        assert_restored()
        assert len(list((root/'eval_weights').glob('step_*'))) == 4
        assert timestamps == {p.name:p.stat().st_mtime_ns for p in snapshot.iterdir()}

        # A committed but corrupted destination is never silently replaced.
        if rank == 0:
            (final/'model_tp_000.pt').write_bytes(b'bad')
        dist.barrier()
        try:
            eval_snapshot.save(final, model, wrapped, 4000, cfg, topology, ema)
        except RuntimeError as error:
            assert 'Truncated evaluation shard' in str(error)
        else:
            raise AssertionError('Corrupt committed snapshot was overwritten')
        assert (final/'model_tp_000.pt').read_bytes() == b'bad'
        assert_restored()
    dist.destroy_process_group()


class EvaluationSnapshotTests(unittest.TestCase):
    def test_distributed_ema_roundtrip_retry_and_immutable_history(self):
        context = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as root:
            workers = [context.Process(target=_worker, args=(root, rank)) for rank in range(4)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(60)
                self.assertTrue(all(not worker.is_alive() for worker in workers), 'Gloo workers hung')
                self.assertEqual([worker.exitcode for worker in workers], [0]*4)
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                    worker.join()

    def test_missing_manifest_is_not_a_committed_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'model_tp_000.pt').write_bytes(b'partial')
            with self.assertRaises(FileNotFoundError):
                eval_snapshot.inspect_snapshot(root)

    def test_restart_retention_preserves_independent_evaluation_history(self):
        from utils.checkpoint_catalog import discover, prune, remove_incomplete
        from tests.test_checkpoint_restart import fixture
        parallel = SimpleNamespace(get_tensor_model_parallel_rank=lambda:0)
        wrapped = SimpleNamespace(use_forward_hook=False, start_param_sync=lambda **kwargs:None)
        config = SimpleNamespace(to_dict=lambda:dict(width=2))
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(eval_snapshot.P, 'state', return_value=parallel), \
             patch.object(eval_snapshot.P, 'tp_size', return_value=1), \
             patch.object(eval_snapshot.P, 'dp_rank', return_value=0), \
             patch.object(eval_snapshot.dist, 'get_rank', return_value=0), \
             patch.object(eval_snapshot.dist, 'get_world_size', return_value=1), \
             patch.object(eval_snapshot.dist, 'broadcast_object_list'), \
             patch.object(eval_snapshot.dist, 'all_gather_object',
                          side_effect=lambda gathered, value:gathered.__setitem__(0, value)):
            root = Path(temporary)
            checkpoints = root/'checkpoints'
            checkpoints.mkdir()
            history = root/'eval_weights'
            for step in range(1, 5):
                fixture(checkpoints, step)
                eval_snapshot.save(history/f'step_{step*1000:07d}', _model(0), wrapped,
                                   step*1000, config, dict(tp=1, dp=1))
            incomplete = checkpoints/'step_0000005'
            incomplete.mkdir()
            (incomplete/'model_tp_000.pt.tmp').write_bytes(b'interrupted checkpoint')
            before = {p.relative_to(history):p.read_bytes() for p in history.rglob('*') if p.is_file()}

            remove_incomplete(checkpoints)
            removed = prune(checkpoints, keep=3)

            self.assertFalse(incomplete.exists())
            self.assertEqual(removed, [str(checkpoints/'step_0000001')])
            self.assertEqual([manifest['step'] for _, manifest in discover(checkpoints)[0]], [4, 3, 2])
            after = {p.relative_to(history):p.read_bytes() for p in history.rglob('*') if p.is_file()}
            self.assertEqual(after, before)
            self.assertEqual([eval_snapshot.inspect_snapshot(path)['step']
                              for path in sorted(history.iterdir())], [1000, 2000, 3000, 4000])

    def test_cli_snapshot_interval_default_and_validation(self):
        from training.train import parse_args
        command = ['train', '--output', '/unused-test-output']
        with patch('sys.argv', command):
            self.assertEqual(parse_args().eval_snapshot_every, 1000)
        with patch('sys.argv', command+['--eval-snapshot-every', '0']):
            self.assertEqual(parse_args().eval_snapshot_every, 0)
        errors = io.StringIO()
        with patch('sys.argv', command+['--eval-snapshot-every', '-1']), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as failure:
                parse_args()
        self.assertEqual(failure.exception.code, 2)
        self.assertIn('eval-snapshot-every >= 0', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
