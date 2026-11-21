"""Failure, retention and full CPU optimizer restore tests."""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.checkpoint_catalog import SCHEMA, discover, durable_json, latest, prune, remove_incomplete, shard_names


def fixture(root, step, world=2, tp=1):
    path = root/f'step_{step:07d}'
    path.mkdir()
    files = {}
    for name in shard_names(world, tp):
        (path/name).write_bytes(b'complete shard')
        files[name] = (path/name).stat().st_size
    durable_json(path/'manifest.json', dict(schema=SCHEMA, step=step, world=world,
        topology=dict(tp=tp, dp=world//tp), config={}, training_state={}, files=files))
    return path


def cpu_restore_worker(root, rank):
    import torch
    import torch.distributed as dist
    import utils.checkpoint as checkpoint_io
    from scripts.validation.compare_data_restart import equal
    dist.init_process_group('gloo', init_method='file://'+str(Path(root)/'gloo'), rank=rank, world_size=2)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    inner = torch.optim.AdamW(model.parameters(), lr=.01)
    optimizer = SimpleNamespace(optimizer=inner)
    old_config = SimpleNamespace(to_dict=lambda:dict(width=3))
    config = SimpleNamespace(to_dict=lambda:dict(width=3, visual_instruction_dropout_probability=0.))
    topology = dict(tp=1, dp=2, pp=1, cp=1, optimizer_instances=1)
    fake_parallel = SimpleNamespace(get_tensor_model_parallel_rank=lambda:0,
        get_data_parallel_group=lambda:dist.group.WORLD)
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
         patch.object(torch.cuda, 'get_rng_state', torch.get_rng_state), \
         patch.object(torch.cuda, 'set_rng_state', torch.set_rng_state):
        path = Path(root)/'step_0000001'
        checkpoint_io.save(path, model, optimizer, 1, old_config, topology, {'schedule':'same'}, data)
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
    def test_old_wan_contract_still_rejects_another_objective_or_source_tree(self):
        from utils.checkpoint import check_resume_contract
        old_state = dict(backbone_adapter_sha256='685214d7b070ca5404251ce3a60413b3c2f3b6809916e074b5aa9fa2a784fbde',
                         data=dict(adapter_sha256='178f411321296755691fa8de032a4c4b9d9f42b7958ced995bc7804007fdb7ef'),
                         seed=42)
        new_state = dict(backbone_adapter_sha256='7ff2690b35a4319311dc03667258dd1033e482bbd20f74b735f49cf335076230',
                         data=dict(adapter_sha256='89b8370775043b6a7e4eca126c16efdb8398977c5d3c24b7ece6335557f930d5'),
                         seed=42)
        config = dict(renderer='wan', visual_instruction_dropout_probability=0.)
        manifest = dict(config=dict(renderer='wan'), training_state=old_state)
        check_resume_contract(manifest, config, new_state)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            check_resume_contract(manifest, dict(config, visual_instruction_dropout_probability=.1), new_state)
        for other in (dict(new_state, seed=43), dict(new_state, backbone_adapter_sha256='different'),
                      dict(new_state, data=dict(adapter_sha256='different'))):
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                check_resume_contract(manifest, config, other)
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            check_resume_contract(dict(manifest, config=config), config, new_state)

    def test_old_formats_are_rejected_before_loading_weights(self):
        import utils.checkpoint as checkpoint_io
        for schema in (3, 4, 5, 6, 'cq_eval_snapshot_v1'):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as temporary:
                path = fixture(Path(temporary), 1)
                manifest = json.loads((path/'manifest.json').read_text())
                manifest['schema'] = schema
                durable_json(path/'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'schema'):
                    checkpoint_io.load(path, None, None, None, None)
                with self.assertRaisesRegex(ValueError, 'schema'):
                    checkpoint_io.initialize_model(path, None, None)

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
