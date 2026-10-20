"""Failure, retention and full CPU optimizer restore tests."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .checkpoint_catalog import discover, durable_json, latest, prune, remove_incomplete, shard_names


def fixture(root, step, world=2, tp=1):
    path = root/f'step_{step:07d}'
    path.mkdir()
    files = {}
    for name in shard_names(world, tp):
        (path/name).write_bytes(b'complete shard')
        files[name] = (path/name).stat().st_size
    durable_json(path/'manifest.json', dict(schema=5, step=step, world=world,
        topology=dict(tp=tp, dp=world//tp), config={}, training_state={}, files=files))
    return path


def cpu_restore_worker(root, rank):
    import torch
    import torch.distributed as dist
    from . import checkpoint as checkpoint_io
    from .compare_data_restart import equal
    dist.init_process_group('gloo', init_method='file://'+str(Path(root)/'gloo'), rank=rank, world_size=2)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    inner = torch.optim.AdamW(model.parameters(), lr=.01)
    optimizer = SimpleNamespace(optimizer=inner)
    config = SimpleNamespace(to_dict=lambda:dict(width=3))
    topology = dict(tp=1, dp=2, pp=1, cp=1, optimizer_instances=1)
    fake_parallel = SimpleNamespace(get_tensor_model_parallel_rank=lambda:0,
        get_data_parallel_group=lambda:dist.group.WORLD)
    fake_npu = SimpleNamespace(get_rng_state=torch.get_rng_state, set_rng_state=torch.set_rng_state)
    data = dict(documents=dict(cursor=31+rank), serial=9)
    def update():
        inner.zero_grad()
        value = model(torch.rand(4, 3)).square().sum()
        value.backward()
        inner.step()
        return value.detach().clone()
    update()
    with patch.object(checkpoint_io.P, 'state', return_value=fake_parallel), \
         patch.object(checkpoint_io.P, 'dp_rank', return_value=rank), \
         patch.object(torch, 'npu', fake_npu, create=True):
        path = Path(root)/'step_0000001'
        checkpoint_io.save(path, model, optimizer, 1, config, topology, {'schedule':'same'}, data)
        expected_loss = update()
        expected_model = copy.deepcopy(model.state_dict())
        expected_optimizer = copy.deepcopy(inner.state_dict())
        with torch.no_grad():
            for p in model.parameters():p.fill_(999)
        inner.state.clear()
        torch.manual_seed(999)
        restored = {}
        step = checkpoint_io.load(path, model, optimizer, config, topology, {'schedule':'same'}, restored)
        assert step == 1 and restored == data
        equal(update(), expected_loss)
        equal(model.state_dict(), expected_model)
        equal(inner.state_dict(), expected_optimizer)
    dist.destroy_process_group()


class CheckpointRestartTests(unittest.TestCase):
    def test_incomplete_missing_and_truncated_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = fixture(root, 1)
            second = fixture(root, 2)
            (second/'rank_00001.pt').unlink()
            third = fixture(root, 3)
            (third/'model_tp_000.pt').write_bytes(b'bad')
            partial = root/'step_0000004'
            partial.mkdir()
            (partial/'rank_00000.pt.tmp').write_bytes(b'interrupted')
            self.assertEqual(latest(root)[0], str(first))
            self.assertEqual(len(latest(root)[1]), 3)
            remove_incomplete(root)
            self.assertFalse(partial.exists())
            self.assertTrue(second.exists())

    def test_corrupt_all_never_silently_starts_over(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = fixture(root, 1)
            (path/'rank_00001.pt').unlink()
            with self.assertRaisesRegex(RuntimeError, 'refusing to restart'):
                latest(root)

    def test_retention_commit_before_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step in range(1, 4):fixture(root, step)
            partial = root/'step_0000004'
            partial.mkdir()
            (partial/'model_tp_000.pt.tmp').write_bytes(b'partial')
            self.assertEqual(len(discover(root)[0]), 3)
            remove_incomplete(root)
            fixture(root, 4)
            prune(root, 3)
            self.assertEqual([m['step'] for _,m in discover(root)[0]], [4,3,2])
            self.assertEqual(len(list(root.iterdir())), 3)

    def test_distributed_model_optimizer_rng_cursor_resume(self):
        context = mp.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary:
            workers = [context.Process(target=cpu_restore_worker,args=(temporary,rank)) for rank in range(2)]
            for worker in workers:worker.start()
            for worker in workers:
                worker.join(40)
                if worker.is_alive():worker.kill();worker.join()
                self.assertEqual(worker.exitcode,0)


if __name__ == '__main__':unittest.main()
