"""CUDA runtime regressions that can be checked without a GPU or NPU wheel."""
import os
from unittest.mock import patch

from distributed.mc2 import configure_mc2


def test_cuda_disables_mc2_before_any_npu_probe_or_collective():
    # Production YAML requests MC2. CUDA must not call npu-smi, allocate an
    # NPU tensor, or enter an HCCL collective even with that unchanged YAML.
    with patch.dict(os.environ, PRETRAIN_MC2_ROW='1', RANK='0'), \
         patch('distributed.accel.IS_NPU', False), \
         patch('subprocess.check_output', side_effect=AssertionError('NPU probe')), \
         patch('torch.tensor', side_effect=AssertionError('NPU allocation')), \
         patch('torch.distributed.all_reduce', side_effect=AssertionError('NPU collective')):
        configure_mc2(2)
        assert os.environ['PRETRAIN_MC2_ROW'] == '0'


def test_unchanged_ascend_yaml_resolves_cuda_runtime_lock():
    from utils.dependencies import lock_path, name

    with patch.dict(os.environ, PRETRAIN_STACK='core17'), patch('distributed.accel.IS_NPU', False):
        assert name() == 'cuda'
        assert lock_path().name == 'deps.cuda.lock.json'


def test_production_synthetic_cli_keeps_model_configuration():
    from training.train import parse_args
    from configs.config import RunConfig

    path = 'configs/qwen35_2b_wan.yaml'
    before = RunConfig.load(path).to_dict()
    with patch.dict(os.environ):
        args = parse_args(['--config', path, '--synthetic', '--tp', '1', '--global-batch', '1',
                           '--log-all-ranks', '--output', '/tmp/cq-test-unused'])
    assert args.synthetic and args.log_all_ranks
    assert args.tp == 1 and args.global_batch == 1
    assert RunConfig.load(path).to_dict() == before
