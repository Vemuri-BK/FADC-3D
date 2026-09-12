import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from fadc_av.run import reusable_inventory
from fadc_av.notebook_runtime import NotebookSession


class RestartTests(unittest.TestCase):
    def test_both_training_cells_with_fresh_python_globals(self):
        for variant in ('enc3', 'all_encoders'):
            notebook = json.loads(Path(f'kaggle_fadc_av_{variant}_full_s42.ipynb').read_text())
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / 'last.pth').write_bytes(b'placeholder')
                state = dict(repo=str(Path.cwd()), config='fadc_av/experiment.json',
                             output=str(output), cache_root='cache', resume='auto')
                state_path = output / 'state.json'
                state_path.write_text(json.dumps(state))
                source = ''.join(notebook['cells'][4]['source'])
                source = source.replace(f'/kaggle/working/fadc_av_{variant}_session.json', state_path.as_posix())
                with patch('subprocess.run') as execute:
                    exec(compile(source, 'training_cell', 'exec'), {})
                args = execute.call_args.args[0]
                self.assertIn('--resume', args)
                self.assertIn(str(output / 'last.pth'), args)
                self.assertEqual(execute.call_count, 1)

    def test_report_refresh_and_timestamp_independent_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {'expected_train': 1, 'expected_val': 1, 'patch_size': [4, 4, 4]}
            for split in ('train', 'val'):
                (root / split).mkdir()
                np.savez(root / split / f'{split}.npz', image=np.ones((2, 4, 4, 4)),
                         label=np.ones((1, 4, 4, 4)))
            report = root / 'preflight.json'
            reusable_inventory(root, cfg, report, refresh=True)
            self.assertTrue(json.loads(report.read_text())['data_checked'])
            self.assertFalse(json.loads(report.read_text())['passed'])
            path = root / 'train/train.npz'
            os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 100))
            with patch('fadc_av.run.np.load', side_effect=AssertionError('Do not decompress unchanged archives')):
                reusable_inventory(root, cfg, report, refresh=True)
            # Same shape and file size, but changed archive CRC: revalidation catches invalid labels.
            np.savez(path, image=np.ones((2, 4, 4, 4)), label=np.full((1, 4, 4, 4), 2.0))
            with self.assertRaisesRegex(ValueError, 'nonbinary'):
                reusable_inventory(root, cfg, report, refresh=True)

    def test_saved_state_resumes_without_notebook_variables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'seed': 42}))
            output = root / 'outputs'
            output.mkdir()
            (output / 'last.pth').write_bytes(b'placeholder')
            state = dict(repo=str(root), config='config.json', output=str(output),
                         cache_root='cache', resume='auto')
            # JSON roundtrip models restoring settings after all kernel variables were lost.
            session = NotebookSession(json.loads(json.dumps(state)))
            with patch.object(session, 'execute') as execute:
                session.train()
            args = execute.call_args.args
            self.assertIn('--resume', args)
            self.assertIn(str(output / 'last.pth'), args)
            self.assertIn('--refresh-preflight', args)
            self.assertEqual(execute.call_count, 1)  # no optimizer preflight on resume
            (output / 'last.pth').unlink()
            (output / 'train_log.json').write_text('[]')
            with self.assertRaisesRegex(FileNotFoundError, 'history exists'):
                session.train()

    def test_legacy_checkpoint_and_explicit_missing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{"seed":42}')
            (root / 'last.pt').write_bytes(b'placeholder')
            state = dict(repo=str(root), config='config.json', output=str(root), cache_root='cache', resume='auto')
            self.assertEqual(NotebookSession(state).checkpoint(), root / 'last.pt')
            state['resume'] = str(root / 'missing.pth')
            with self.assertRaises(FileNotFoundError):
                NotebookSession(state).checkpoint()
