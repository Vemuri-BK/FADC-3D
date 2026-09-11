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


cells = [cell("markdown", """# FADC-AV: first image-only 3D experiment
Existing two-channel 3D U-Net; replace enc3's second convolution only.
Four frequency components, dilations (1,2,3), AdaKern enabled, no deep supervision.
Attach the preprocessed two-channel MAMA-MIA cache and enable a GPU + Internet.
Set EXPECTED_COMMIT to the full commit containing these files after publishing the branch.
Run cells in order. A failed check stops execution; do not silently reduce batch size.
The historical baseline is a reference; corrected augmentation seeding means a new matched
baseline is needed for a strict final comparison. Validation is model selection, not a held-out test.
"""), cell("code", """from pathlib import Path
import os, re, subprocess, sys, json
REPO_URL = 'https://github.com/Vemuri-BK/FADC-3D.git'
BRANCH = 'feature/fadc-AV'
EXPECTED_COMMIT = ''  # Required: full 40-character SHA containing fadc_av
CACHE_ROOT = Path('/kaggle/input/datasets/bharathvemurik/mama-mia-preprocessed-cache-2ch')
REPO = Path('/kaggle/working/FADC-AV')
OUTPUT = Path('/kaggle/working/outputs/fadc_av_enc3_full_s42')
RESUME = ''  # Optional trusted last.pt from a previous Kaggle session
assert re.fullmatch(r'[0-9a-fA-F]{40}', EXPECTED_COMMIT), 'Set the published full commit SHA first'
assert (CACHE_ROOT / 'train').is_dir() and (CACHE_ROOT / 'val').is_dir(), 'Correct CACHE_ROOT to contain train/ and val/'
def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)
if not REPO.exists():
    run('git', 'clone', '--branch', BRANCH, REPO_URL, REPO)
assert not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip(), 'Checkout has local changes'
run('git', '-C', REPO, 'fetch', 'origin', BRANCH)
run('git', '-C', REPO, 'checkout', '--detach', EXPECTED_COMMIT)
assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == EXPECTED_COMMIT.lower()
os.chdir(REPO)
assert (REPO / 'fadc_av/run.py').is_file()
"""), cell("code", """# Keep Kaggle's CUDA-compatible PyTorch installation.
run(sys.executable, '-m', 'pip', 'install', 'monai==1.5.2', 'nibabel', 'pandas', 'tqdm')
import torch
assert torch.cuda.is_available(), 'Enable GPU accelerator'
print(torch.__version__, torch.cuda.get_device_name(0))
OUTPUT.mkdir(parents=True, exist_ok=True)
(OUTPUT / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
env = dict(os.environ, PYTHONHASHSEED='42', CUDA_VISIBLE_DEVICES='0')
run(sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'fadc_av/tests', '-v', env=env)
"""), cell("code", """# Scan every cache file, check disjoint patient IDs, then test the production model/patch/batch.
common = [sys.executable, '-u', '-m', 'fadc_av.run', '--config', 'fadc_av/experiment.json',
          '--cache-root', str(CACHE_ROOT), '--output', str(OUTPUT)]
run(*common, '--mode', 'preflight', env=env)
report = json.loads((OUTPUT / 'preflight.json').read_text())
assert report['passed']
print('Preflight passed; peak allocated GPU GB:', report['peak_gpu_gb'])
"""), cell("code", """# Full training starts only after this session's successful preflight.
assert report['passed'] and report['config'] == json.loads(Path('fadc_av/experiment.json').read_text())
resume_args = ['--resume', RESUME] if RESUME else []
run(*common, '--mode', 'train', *resume_args, env=env)
"""), cell("code", """# Re-evaluate the selected checkpoint through the same whole-volume evaluator.
run(*common, '--mode', 'evaluate', '--resume', OUTPUT / 'best.pt', env=env)
from IPython.display import FileLink, display
for name in ('best.pt', 'last.pt', 'train_log.json', 'evaluation.json', 'train_provenance.json', 'pip_freeze.txt'):
    display(FileLink(str(OUTPUT / name)))
print('Save a Kaggle notebook version with outputs to retain artifacts after the session ends.')
""")]
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--commit', default='', help='Published full SHA containing the experiment code')
args = parser.parse_args()
if args.commit:
    if not re.fullmatch(r'[0-9a-fA-F]{40}', args.commit):
        parser.error('--commit must be a full 40-character Git SHA')
    for item in cells:
        item['source'] = [line.replace("EXPECTED_COMMIT = ''", f"EXPECTED_COMMIT = '{args.commit.lower()}'")
                          for line in item['source']]
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
Path('kaggle_fadc_av_enc3_full_s42.ipynb').write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
