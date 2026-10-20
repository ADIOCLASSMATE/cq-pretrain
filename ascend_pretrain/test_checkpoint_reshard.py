import unittest
import torch
from .checkpoint_reshard import reshard_tensor,reshard_state
from .parallel import mark_sharded


class ReshardContracts(unittest.TestCase):
    def test_exact_fused_and_row_partition_conversions(self):
        for dim,shape,logical in [(0,(96,7),(48,48)),(0,(96,7),(32,16,24,24)),
                                  (0,(96,7),(32,32,32)),(1,(7,96),(96,)),(0,(96,),(96,))]:
            full=torch.arange(torch.tensor(shape).prod()).reshape(shape).to(torch.bfloat16)
            def oracle(size,rank):
                return torch.cat([piece.chunk(size,dim)[rank] for piece in full.split(logical,dim)],dim)
            for source_tp,target_tp in [(2,4),(4,2),(2,8),(8,1),(1,4)]:
                saved=[oracle(source_tp,r) for r in range(source_tp)]
                for rank in range(target_tp):
                    target=torch.nn.Parameter(torch.empty_like(oracle(target_tp,rank)))
                    mark_sharded(target,dim=dim,splits=logical)
                    actual=reshard_tensor(saved,target,target_tp,rank)
                    torch.testing.assert_close(actual,oracle(target_tp,rank),rtol=0,atol=0)
                    if len(set(logical))==1:
                        del target._checkpoint_splits;target.partition_stride=len(logical)
                        torch.testing.assert_close(reshard_tensor(saved,target,target_tp,rank),oracle(target_tp,rank),rtol=0,atol=0)

    def test_replicated_divergence_and_invalid_shapes_are_rejected(self):
        target=torch.nn.Parameter(torch.arange(16,dtype=torch.float32))
        torch.testing.assert_close(reshard_tensor([target.detach().clone()]*2,target,4,3),target)
        with self.assertRaises(ValueError):reshard_tensor([target.detach(),target.detach()+1],target,4,0)
        mark_sharded(target)
        with self.assertRaises(ValueError):reshard_tensor([torch.zeros(16)]*2,target,4,0)

    def test_state_dict_keeps_parameter_partition_metadata(self):
        model=torch.nn.Linear(3,4,bias=False).to(torch.bfloat16)
        mark_sharded(model.weight,dim=0)
        full=torch.arange(48).reshape(16,3).to(torch.bfloat16)
        saved=[{'weight':p} for p in full.chunk(2)]
        state=reshard_state(saved,model,4,2)
        model.load_state_dict(state)
        torch.testing.assert_close(model.weight,full[8:12],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
