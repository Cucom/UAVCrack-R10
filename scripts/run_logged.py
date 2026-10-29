"""Launch an existing command with durable logs, exit status and sampled NVML memory."""
import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--timeout-seconds', type=float, default=0)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command[:1] == ['--'] else a.command
if not command:
    p.error('command required after --')
log = Path('logs') / (a.name + '.log')
status_path = Path('logs') / (a.name + '_status.json')
memory_path = Path('logs') / (a.name + '_nvml.csv')
if log.exists() or status_path.exists() or memory_path.exists():
    p.error('run name already exists; select a new name to preserve evidence')
log.parent.mkdir(exist_ok=True)
status = dict(command=command, cwd=str(Path.cwd()), start_utc=datetime.now(timezone.utc).isoformat(),
              state='running', exit_code=None)
status_path.write_text(json.dumps(status, indent=2)+'\n')
stop = threading.Event()
def monitor():
    with memory_path.open('w') as f:
        w = csv.writer(f)
        w.writerow(['unix_time','device_index','memory_used_mib','gpu_utilization_percent'])
        while not stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                                   capture_output=True,text=True,timeout=3)
                if r.returncode == 0:
                    for line in r.stdout.strip().splitlines():
                        w.writerow([time.time(), *[v.strip() for v in line.split(',')]])
                    f.flush()
            except (subprocess.SubprocessError,OSError):
                pass
            stop.wait(.5)
thread = threading.Thread(target=monitor,daemon=True)
thread.start()
start = time.perf_counter()
try:
    with log.open('w') as f:
        proc = subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code = proc.wait(timeout=a.timeout_seconds or None)
        except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
            status['interruption'] = type(exc).__name__
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            code = 124 if isinstance(exc,subprocess.TimeoutExpired) else 130
except OSError as exc:
    code = 127
    status['launch_error'] = str(exc)
finally:
    elapsed = time.perf_counter() - start
    stop.set()
    thread.join(timeout=4)
status.update(state='finished' if code == 0 else 'failed',exit_code=code,elapsed_seconds=elapsed,
              end_utc=datetime.now(timezone.utc).isoformat())
status_path.write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2),flush=True)
raise SystemExit(code)
