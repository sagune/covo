"""Stop only verified original workers, preserving interruption evidence."""
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE=Path(__file__).resolve().parent
parent,worker=map(int,sys.argv[1:])
commands={pid:Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode() for pid in (parent,worker)}
assert str(HERE/'reader.py') in commands[worker] and commands[worker].rstrip().endswith(' train')
assert 'reader.py pipeline' in commands[parent]
assert f'PPid:\t{parent}\n' in Path(f'/proc/{worker}/status').read_text()
os.kill(parent,signal.SIGTERM); os.kill(worker,signal.SIGTERM)
for _ in range(100):
    if not Path(f'/proc/{worker}').exists(): break
    time.sleep(.1)
last=json.loads((HERE/'R/status.json').read_text())
record=dict(reason='User requested higher utilization; execution-only benchmark then identical-schedule replay/resume',
            stopped=time.strftime('%Y-%m-%dT%H:%M:%S%z'),pids=commands,last_completed_status=last,
            existing_complete_checkpoints=[str(p) for p in (HERE/'R/checkpoints').glob('*/metadata.json')],
            policy='Preserve all original source/protocol/logs; never initialize formal training from smoke weights')
path=HERE/f'interrupted_for_throughput_{time.time_ns()}.json'
with path.open('x') as stream: json.dump(record,stream,indent=2)
print(json.dumps(record))
