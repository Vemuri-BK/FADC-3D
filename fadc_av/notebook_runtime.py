"""Notebook orchestration using saved settings instead of kernel variables."""
import json
import os
from pathlib import Path
import subprocess
import sys


class NotebookSession:
    def __init__(self, state):
        self.state = state
        self.repo = Path(state['repo'])
        self.output = Path(state['output'])
        self.config = self.repo / state['config']
        self.output.mkdir(parents=True, exist_ok=True)
        self.cfg = json.loads(self.config.read_text())
        self.env = dict(os.environ, PYTHONHASHSEED=str(self.cfg['seed']), CUDA_VISIBLE_DEVICES='0')
        self.common = [sys.executable, '-u', '-m', 'fadc_av.run', '--config', str(self.config),
                       '--cache-root', state['cache_root'], '--output', str(self.output)]

    def execute(self, *args):
        subprocess.run([str(x) for x in args], check=True, cwd=self.repo, env=self.env)

    def checkpoint(self, kind='last'):
        if kind == 'last' and self.state.get('resume', 'auto') not in ('auto', ''):
            candidate = Path(self.state['resume'])
            if not candidate.is_file():
                raise FileNotFoundError(f'Resume checkpoint missing: {candidate}')
            return candidate
        found = next((self.output / f'{kind}.{ext}' for ext in ('pth', 'pt')
                      if (self.output / f'{kind}.{ext}').is_file()), None)
        if kind == 'last' and self.state.get('resume', 'auto') == '':
            if found:
                raise ValueError('Fresh run requested but checkpoint exists; set RESUME="auto"')
            return None
        return found

    def prepare(self):
        if self.checkpoint() is not None:
            print('Checkpoint found: optimizer preflight is unnecessary. Training will verify data and resume.')
            return
        if any((self.output / f).exists() for f in ('train_log.json', 'best.pth', 'best.pt')):
            raise FileNotFoundError('Run history exists but last checkpoint is missing. Restore last.pth; do not restart silently.')
        try:
            report = json.loads((self.output / 'preflight.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            report = {}
        if report.get('passed') is True and report.get('config') == self.cfg:
            print('Successful model preflight found on disk. Data freshness will be checked at training startup.')
            return
        self.execute(*self.common, '--mode', 'preflight')

    def train(self):
        self.prepare()
        checkpoint = self.checkpoint()
        print(f'Resuming {checkpoint}' if checkpoint else 'Starting a new experiment')
        resume = ['--resume', str(checkpoint)] if checkpoint else []
        self.execute(*self.common, '--mode', 'train', '--preflight-report', self.output / 'preflight.json',
                     '--refresh-preflight', *resume)

    def evaluate(self):
        checkpoint = self.checkpoint('best')
        if checkpoint is None:
            raise FileNotFoundError('No best checkpoint yet. Complete a scheduled validation first.')
        self.execute(*self.common, '--mode', 'evaluate', '--resume', checkpoint,
                     '--preflight-report', self.output / 'preflight.json', '--refresh-preflight')

    def verify(self):
        self.execute(sys.executable, '-m', 'pip', 'install', 'monai==1.5.2', 'nibabel', 'pandas', 'tqdm')
        self.execute(sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "Enable GPU"; print(torch.__version__, torch.cuda.get_device_name(0))')
        packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (self.output / 'pip_freeze.txt').write_text(packages)
        self.execute(sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'fadc_av/tests', '-v')
