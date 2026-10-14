"""Exercise queue handoff without loading models or touching the real GPU/run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import overnight as n


class QueueTests(unittest.TestCase):
    def scenario(self,status,expired=False,gpu_busy=False):
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); out=root/'next'; pilot=root/'pilot'
            out.mkdir(); pilot.mkdir(); (pilot/'A_512').mkdir()
            (pilot/'A_512'/'connector.pt').touch()
            (pilot/'pipeline.lock').touch()
            (pilot/'status.json').write_text(json.dumps({'status':status}))
            def fake_run(args,**kwargs):
                calls.append(args[-1])
                if args[-1]=='export': (out/'export_complete.json').write_text('{}')
            with patch.object(n,'OUT',out),patch.object(n,'PILOT',pilot),patch.object(n,'audit',return_value={}), \
                 patch.object(n.time,'time',return_value=n.DEADLINE+(1 if expired else -100)), \
                 patch.object(n.time,'sleep'),patch.object(n.subprocess,'run',side_effect=fake_run), \
                 patch.object(n.subprocess,'check_output',return_value='999' if gpu_busy else ''):
                error=None
                try: n.supervisor()
                except RuntimeError as exc: error=str(exc)
            state=json.loads((out/'status.json').read_text())
        return calls,state,error

    def test_clean_negative_pilot_starts_independent_continuation(self):
        calls,state,error=self.scenario('COMPLETED_NEGATIVE_STAGE_A')
        self.assertEqual(calls,['export','train'])
        self.assertIsNone(error)

    def test_technical_failure_is_not_treated_as_negative_scientific_result(self):
        calls,state,error=self.scenario('FAILED')
        self.assertEqual(calls,[])
        self.assertEqual(state['status'],'FAILED')
        self.assertIn('failed technically',error)

    def test_deadline_starts_no_extra_jobs(self):
        calls,state,error=self.scenario('RUNNING',expired=True)
        self.assertEqual(calls,[])
        self.assertEqual(state['status'],'DEADLINE_BEFORE_CONTINUATION')
        self.assertIsNone(error)

    def test_busy_gpu_not_preempted(self):
        calls,state,error=self.scenario('COMPLETED',gpu_busy=True)
        self.assertEqual(calls,[])
        self.assertIn('Other GPU process',error)


if __name__=='__main__': unittest.main(verbosity=2)
