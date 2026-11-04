"""Independent linear-chain references for the optional training adapter."""
import unittest
import torch
from models.mc2 import _RowAllReduce


class MC2GradientTests(unittest.TestCase):
    def test_fragmented_topology_cannot_use_mc2(self):
        from models.mc2 import topology_supports_mc2
        mapping='NPU ID Chip ID Chip Logic ID\n1 0 0 Ascend\n1 1 - Mcu\n10 0 1 Ascend'
        topo='NPU1 NPU10 CPU Affinity\nNPU1 X PHB 0-47\nNPU10 PHB X 0-47'
        self.assertFalse(topology_supports_mc2(2,0,2,mapping,topo))
        self.assertTrue(topology_supports_mc2(2,1,2,mapping,topo.replace('PHB','HCCS')))
        self.assertFalse(topology_supports_mc2(4,0,2,mapping,topo))

    def test_local_chain_rule(self):
        torch.manual_seed(94)
        x=torch.randn(2,3,8,dtype=torch.float64).transpose(0,1).requires_grad_()
        w=torch.randn(12,8,dtype=torch.float64,requires_grad=True)
        r=x.detach().clone().requires_grad_();v=w.detach().clone().requires_grad_()
        g=torch.randn(3,2,12,dtype=torch.float64)
        y=_RowAllReduce.apply(x,w,None);z=torch.nn.functional.linear(r,v)
        y.backward(g);z.backward(g)
        for a,b in ((y,z),(x.grad,r.grad),(w.grad,v.grad)):torch.testing.assert_close(a,b)

    def test_fp32_buffer_and_hook_gradient(self):
        torch.manual_seed(53)
        w=torch.randn(12,8,dtype=torch.float64,requires_grad=True)
        w.main_grad=torch.zeros_like(w,dtype=torch.float32)
        w.grad_added_to_main_grad=False;w.zero_out_wgrad=True
        reference=w.detach().clone().requires_grad_()
        for _ in range(2):
            x=torch.randn(17,8,dtype=torch.float64,requires_grad=True)
            probe=torch.randn(17,12,dtype=torch.float64)
            _RowAllReduce.apply(x,w,None).backward(probe)
            torch.nn.functional.linear(x.detach(),reference).backward(probe)
        torch.testing.assert_close(w.main_grad,reference.grad.float(),rtol=2e-6,atol=2e-6)
        self.assertTrue(w.grad_added_to_main_grad)
        self.assertEqual(w.grad.count_nonzero(),0)


if __name__=='__main__':unittest.main()
