"""Optional branches must enqueue identical collective sequences on every rank."""
from types import SimpleNamespace
import unittest
from .input_pipeline.parallel import OrderedParameterGathers


class CollectiveOrderTests(unittest.TestCase):
    def test_checkpoint_forces_fresh_gather_of_updated_weights(self):
        calls=[]
        group=SimpleNamespace(ddp_config=SimpleNamespace(overlap_param_gather=True),
                              param_gather_dispatched=True,param_gather_handle=None)
        group.start_param_sync=lambda force_sync=False:calls.append(force_sync)
        OrderedParameterGathers(SimpleNamespace(bucket_groups=[group],expert_parallel_bucket_groups=[]))
        group.start_param_sync()  # the forward already gathered this group
        self.assertEqual(calls,[])
        group.start_param_sync(force_sync=True)  # optimizer shards have changed
        self.assertEqual(calls,[True])

    def test_skipped_vision_and_repeated_queries_preserve_order(self):
        def simulate(accesses):
            calls=[];groups=[]
            for index in range(6):
                group=SimpleNamespace(ddp_config=SimpleNamespace(overlap_param_gather=True),
                    param_gather_dispatched=False,param_gather_handle=None)
                def start(force_sync=False,index=index,group=group):
                    self.assertFalse(group.param_gather_dispatched)
                    calls.append(index);group.param_gather_dispatched=True
                group.start_param_sync=start;groups.append(group)
            OrderedParameterGathers(SimpleNamespace(bucket_groups=list(reversed(groups)),
                                                    expert_parallel_bucket_groups=[]))
            for access in accesses:groups[access].start_param_sync()
            return calls
        self.assertEqual(simulate([0,1,2,3,0,4,5]),list(range(6)))
        self.assertEqual(simulate([0,3,0,5]),list(range(6)))
        self.assertEqual(simulate([5,0,2,5]),list(range(6)))


if __name__=='__main__':unittest.main()
