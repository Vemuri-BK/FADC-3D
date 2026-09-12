"""Generate the thin Kaggle launcher; no training logic is embedded here."""
import argparse
import json
import re
from pathlib import Path


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


BOOTSTRAP = """from pathlib import Path
import json, importlib.util
STATE_FILE = Path('/kaggle/working/fadc_av_enc3_session.json')
assert STATE_FILE.is_file(), 'Run the setup cell once to restore session settings.'
state = json.loads(STATE_FILE.read_text())
spec = importlib.util.spec_from_file_location('fadc_notebook_runtime', Path(state['repo']) / 'fadc_av/notebook_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
session = runtime.NotebookSession(state)
"""

cells = [cell('markdown', """# FADC-AV: first image-only 3D experiment
Existing two-channel 3D U-Net; replace enc3's second convolution only.
Four bands, dilation (1,2,3), AdaKern; full validation on all 306 cases every 10 epochs.
Enable GPU and Internet. Attach the SAME dataset version used previously.
Setup saves settings on disk. Later cells reload them and do not depend on old Python variables.
RESUME='auto' resumes last.pth (or legacy last.pt); an explicit path can restore an attached checkpoint.
Keep both last.pth and best.pth when exporting checkpoints. If Kaggle did not preserve working files,
attach saved outputs and set RESUME to last.pth. Checkpoints cannot be recovered from displayed logs.
On restart: run setup and environment checks, then training. Model preflight is skipped when resuming.
Changed timestamps alone do not invalidate new archive signatures. Older reports may need one automatic
array validation scan, with progress; training then continues without manual cell changes.
"""), cell('code', """from pathlib import Path
import json, subprocess, tempfile
REPO_URL = 'https://github.com/Vemuri-BK/FADC-3D.git'
BRANCH = 'feature/fadc-AV'
EXPECTED_COMMIT = ''
CACHE_ROOT = Path('/kaggle/input/datasets/vbk1999/mama-mia-preprocessed-cache-2ch')
OUTPUT = Path('/kaggle/working/outputs/fadc_av_enc3_full_s42')
RESUME = 'auto'  # Or a trusted last.pth path from saved outputs; keep best.pth beside it.
STATE_FILE = Path('/kaggle/working/fadc_av_enc3_session.json')
CONFIG_RELATIVE = 'fadc_av/experiment.json'
assert len(EXPECTED_COMMIT) == 40, 'Set the pinned commit SHA'
assert (CACHE_ROOT / 'train').is_dir() and (CACHE_ROOT / 'val').is_dir(), 'Correct CACHE_ROOT'
def git(*args):
    return subprocess.check_output(['git', *map(str, args)], text=True).strip()
def usable(path):
    try:
        return (git('-C', path, 'remote', 'get-url', 'origin') == REPO_URL
                and git('-C', path, 'rev-parse', 'HEAD') == EXPECTED_COMMIT
                and not git('-C', path, 'status', '--porcelain'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
REPO = Path('/kaggle/working') / ('FADC-AV-' + EXPECTED_COMMIT[:12])
if STATE_FILE.is_file():
    try:
        previous = json.loads(STATE_FILE.read_text())
        if usable(Path(previous['repo'])):
            REPO = Path(previous['repo'])
    except (KeyError, ValueError):
        pass
if not (REPO.exists() and usable(REPO)):
    if REPO.exists():
        print('Preserving existing checkout; creating a clean code folder.')
        REPO = Path(tempfile.mkdtemp(prefix='FADC-AV-clean-', dir='/kaggle/working'))
    subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(REPO)], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', EXPECTED_COMMIT], check=True)
assert usable(REPO)
config = json.loads((REPO / CONFIG_RELATIVE).read_text())
assert config['variant'] == 'fadc_enc3', 'Wrong experiment configuration'
state = dict(repo=str(REPO), config=CONFIG_RELATIVE, cache_root=str(CACHE_ROOT),
             output=str(OUTPUT), resume=RESUME, commit=EXPECTED_COMMIT)
STATE_FILE.write_text(json.dumps(state, indent=2))
print('Setup saved:', STATE_FILE)
print('Experiment:', config['variant'], '| output:', OUTPUT, '| resume:', RESUME)
"""), cell('code', BOOTSTRAP + "\nsession.verify()\n"),
cell('code', BOOTSTRAP + "\n# Optional for resume: prepare() detects existing checkpoints.\nsession.prepare()\n"),
cell('code', BOOTSTRAP + "\n# Reads saved reports and checkpoints; no report/common/env variables needed.\nsession.train()\n"),
cell('code', BOOTSTRAP + """
session.evaluate()
from IPython.display import FileLink, display
for name in ('best.pth', 'last.pth', 'train_log.json', 'train_log.csv', 'evaluation.json'):
    path = session.output / name
    if path.is_file():
        display(FileLink(str(path)))
print('Save/download outputs before ending the Kaggle session.')
""")]
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--commit', default='', help='Published full SHA containing the experiment code')
parser.add_argument('--variant', choices=('enc3', 'all_encoders'), default='enc3')
args = parser.parse_args()
if args.variant == 'all_encoders':
    replacements = {
        'first image-only 3D experiment': 'all-encoder image-only 3D experiment',
        "replace enc3's second convolution only.": 'replace both convolutions in enc1, enc2, enc3 and enc4 (eight FADC convolutions).',
        'fadc_av/experiment.json': 'fadc_av/experiment_all_encoders.json',
        'fadc_av_enc3_full_s42': 'fadc_av_all_encoders_full_s42',
        'fadc_av_enc3_session.json': 'fadc_av_all_encoders_session.json',
        "config['variant'] == 'fadc_enc3'": "config['variant'] == 'fadc_all_encoders'",
    }
    for item in cells:
        source = ''.join(item['source'])
        for old, new in replacements.items():
            source = source.replace(old, new)
        item['source'] = source.splitlines(keepends=True)
if args.commit:
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.commit):
        parser.error('--commit must be a full 40-character Git SHA')
    for item in cells:
        item['source'] = [line.replace("EXPECTED_COMMIT = ''", f"EXPECTED_COMMIT = '{args.commit.lower()}'")
                          for line in item['source']]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
name = 'all_encoders' if args.variant == 'all_encoders' else 'enc3'
Path(f'kaggle_fadc_av_{name}_full_s42.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
