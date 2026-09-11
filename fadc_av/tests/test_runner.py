"""Small synthetic-volume integration checks; no clinical accuracy claim."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from fadc_av import run


class RunnerTests(unittest.TestCase):
    def test_train_resume_evaluate_and_split_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = json.loads(Path('fadc_av/experiment.json').read_text())
            cfg.update(base_filters=2, patch_size=[32, 32, 16], epochs=2, val_every=1,
                       warmup_epochs=0, num_workers=0, expected_train=2, expected_val=1)
            config = root / 'config.json'
            config.write_text(json.dumps(cfg))
            for split, count in [('train', 2), ('val', 1)]:
                (root / split).mkdir()
                for i in range(count):
                    x = np.random.default_rng(i).random((2, 32, 32, 16), dtype=np.float32)
                    y = np.zeros((1, 32, 32, 16), dtype=np.uint8)
                    y[:, 10:18, 10:18, 4:10] = 1
                    np.savez(root / split / f'{split}_{i}.npz', image=x, label=y)
            original_save = run.atomic_save
            def save_with_epoch_one(state, path):
                original_save(state, path)
                if path.name == 'last.pt' and state['epoch'] == 1:
                    original_save(state, root / 'epoch_one.pt')
                    original_save(state, root / 'best.pt')
            def launch(mode, output, resume=None):
                argv = ['run', '--config', str(config), '--cache-root', str(root),
                        '--output', str(output), '--mode', mode, '--allow-cpu']
                if resume:
                    argv += ['--resume', str(resume)]
                with patch('sys.argv', argv), patch('torch.cuda.is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    run.main()
            output = root / 'uninterrupted'
            launch('preflight', output)
            self.assertTrue(json.loads((output / 'preflight.json').read_text())['passed'])
            with patch.object(run, 'atomic_save', side_effect=save_with_epoch_one):
                launch('train', output)
            resumed = root / 'resumed'
            launch('train', resumed, root / 'epoch_one.pt')
            full = torch.load(output / 'last.pt', weights_only=False)
            continued = torch.load(resumed / 'last.pt', weights_only=False)
            for name, value in full['model'].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, continued['model'][name], rtol=0, atol=0)
            launch('evaluate', resumed, resumed / 'best.pt' if (resumed / 'best.pt').exists() else output / 'best.pt')
            self.assertTrue((resumed / 'evaluation.json').is_file())
            # An overlapping filename must be rejected even with correct counts.
            (root / 'val' / 'val_0.npz').rename(root / 'val' / 'train_0.npz')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                run.inventory(root, cfg)

    def test_notebook_python_cells_compile(self):
        notebook = json.loads(Path('kaggle_fadc_av_enc3_full_s42.ipynb').read_text())
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell_{i}', 'exec')


if __name__ == '__main__':
    unittest.main()
