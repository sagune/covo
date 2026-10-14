"""CPU-only queue safety tests; do not touch actual experiment outputs."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import serial_queue as q

class Tests(unittest.TestCase):
    def test_identity_existing_and_missing(self):
        actual=q.identity(os.getpid())
        self.assertIsInstance(actual['start_ticks'],int)
        self.assertIn('python',actual['cmdline'])
        self.assertIsNone(q.identity(999999999))

    def test_source_mutation_rejected(self):
        with tempfile.TemporaryDirectory(prefix='evidence_queue_cpu_test_') as d:
            f=Path(d)/'fixture';f.write_text('original')
            spec={'sources':{str(f):q.sha(f)}}
            q.check_sources(spec)
            f.write_text('changed')
            with self.assertRaises(AssertionError):q.check_sources(spec)

    def test_parent_failure_no_new_job(self):
        spec=dict(parent='/not-real-fixture',parent_pid=42,parent_identity={'fixture':True},jobs=[],deadline_epoch=0)
        with patch.object(q,'check_sources'),patch.object(q,'status'),patch.object(q,'identity',return_value={'fixture':True}),patch.object(q,'load',return_value={'stage':'FAILED'}),patch.object(q.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(AssertionError,'failed'):q.run(spec)
            popen.assert_not_called()

    def test_expired_window_no_new_job(self):
        with tempfile.TemporaryDirectory(prefix='evidence_queue_cpu_test_') as d:
            spec=dict(parent='/fixture',parent_pid=42,parent_identity={},jobs=[{'name':'must_not_start'}],deadline_epoch=0)
            with patch.object(q,'HERE',Path(d)),patch.object(q,'check_sources'),patch.object(q,'status') as status,patch.object(q,'identity',return_value=None),patch.object(q,'completed',return_value={'fixture':'complete'}),patch.object(q.subprocess,'Popen') as popen:
                q.run(spec);popen.assert_not_called()
                self.assertEqual(status.call_args.args[0],'WINDOW_REACHED')

if __name__=='__main__':unittest.main()
