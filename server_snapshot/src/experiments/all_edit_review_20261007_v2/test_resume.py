"""CPU tests for checkpoint selection/atomic visibility and log reconstruction."""
import unittest


def effective_logs(attempts):
    result=[]
    for cursor,records in attempts:result=[x for x in result if x['step']<=cursor]+records
    return result


class Tests(unittest.TestCase):
    def test_rollback_to_complete_checkpoint(self):
        old=[{'step':i,'attempt':1} for i in range(1,930)]
        new=[{'step':i,'attempt':2} for i in range(897,1001)]
        got=effective_logs([(0,old),(896,new)])
        self.assertEqual([x['step'] for x in got],list(range(1,1001)))
        self.assertEqual(got[895]['attempt'],1);self.assertEqual(got[896]['attempt'],2)

    def test_fresh_run(self):
        self.assertEqual(effective_logs([(0,[{'step':1},{'step':2}])]),[{'step':1},{'step':2}])

    def test_partial_not_a_resume_candidate(self):
        names=['checkpoint_128','checkpoint_256.partial','smoke_checkpoint','final_checkpoint','checkpoint_384']
        kept=[x for x in names if x.startswith('checkpoint_') and x.split('_')[-1].isdigit()]
        self.assertEqual(kept,['checkpoint_128','checkpoint_384'])


if __name__=='__main__':unittest.main()
