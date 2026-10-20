"""Direct optimizer-buffer accumulation must match dense CE over all chunks."""
import unittest
import torch
from .loss import linear_cross_entropy


class DirectCEAccumulation(unittest.TestCase):
    def test_multiple_microbatches_keep_fp32_gradient_and_hooks(self):
        torch.manual_seed(42);torch.set_num_threads(1)
        weight=torch.nn.Parameter(torch.randn(513,64));reference=weight.detach().clone().requires_grad_()
        weight._ce_direct_accumulation=True;weight.grad_added_to_main_grad=False
        weight.main_grad=torch.zeros_like(weight);calls=[]
        def hook(parameter):
            self.assertTrue(parameter.grad_added_to_main_grad)
            parameter.grad=None;calls.append(True)
        weight.register_post_accumulate_grad_hook(hook)
        for count,scale in ((17,.3),(23,.7)):
            x=torch.randn(count,64,requires_grad=True);other=x.detach().clone().requires_grad_()
            targets=torch.randint(513,(count,))
            actual=linear_cross_entropy(x,weight,targets,7)*scale
            expected=torch.nn.functional.cross_entropy(other@reference.T,targets)*scale
            actual.backward();expected.backward()
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(x.grad,other.grad,atol=2e-6,rtol=2e-5)
        self.assertEqual(len(calls),2)
        torch.testing.assert_close(weight.main_grad,reference.grad,atol=2e-6,rtol=2e-5)


if __name__=='__main__':unittest.main()
