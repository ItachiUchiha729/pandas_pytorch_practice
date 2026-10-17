"""Run a notebook child process with live output and nonempty failure messages."""
from collections import deque
from pathlib import Path
import json
import signal
import subprocess
import tempfile


def run_logged(command, project, log_dir=None):
    project = Path(project)
    log_dir = Path(log_dir) if log_dir else project / 'outputs' / 'presentations' / 'process_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='strategy_', suffix='.log', dir=log_dir)
    import os
    os.close(fd)
    path = Path(name)
    print(f'Live strategy log: {path}', flush=True)
    tail, artifacts = deque(maxlen=35), None
    with path.open('w') as log:
        log.write('Command: ' + repr(command) + '\n')
        log.flush()
        try:
            process = subprocess.Popen(command, cwd=project, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, bufsize=1)
        except OSError as error:
            log.write(str(error)+'\n')
            raise RuntimeError(f'Could not launch strategy: {error}. Log: {path}') from error
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                print(line, end='', flush=True)
                if line.startswith('Artifacts: '):
                    artifacts = Path(line.removeprefix('Artifacts: ').strip())
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        if code < 0:
            reason = f'terminated by {signal.Signals(-code).name} (return code {code})'
            if code == -signal.SIGKILL:
                reason += '; memory pressure or an external kill is possible, not proven by this exit code'
        else:
            reason = f'exited with return code {code}'
        details = '\n'.join(tail) or '(No output was produced before termination.)'
        raise RuntimeError(f'Strategy {reason}.\nPersistent log: {path}\nRun directory: {artifacts}\n{details}')
    if artifacts is None or not (artifacts/'manifest.json').exists():
        raise RuntimeError(f'Strategy exited successfully but did not report a run manifest. Log: {path}')
    if not json.loads((artifacts/'manifest.json').read_text()).get('complete'):
        raise RuntimeError(f'Strategy exited with an incomplete manifest: {artifacts}. Log: {path}')
    return artifacts
