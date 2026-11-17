# Project 2 training

The notebook defaults to `loss="composite"`. It trains the NAF-U-Net with RGB
Charbonnier, `0.10 × (1 − SSIM)`, `0.20 ×` local exposure loss, and `0.02 ×`
VGG-16 feature loss. The exposure term matches average luma in 16×16 and 64×64
regions. The feature term matches the `relu1_2` and `relu2_2` activations of a
frozen ImageNet VGG-16. Training downloads the VGG-16 weights. The notebook
downloads the 100 RAISE RAW images and the LPIPS AlexNet weights as needed.

## Report and saved results

- [Final report](raport.pdf) and [editable LaTeX](raport_latex/main.tex). The
  source includes its figures. To rebuild, run `pdflatex main.tex` twice from
  `raport_latex/`.
- [Training notebook](low_light.ipynb) with the saved Colab outputs and metrics.
- `results/composite/` contains the epoch 29 checkpoint (`best.pt`), the data
  split, the 50-epoch training history, and the notebook's test metrics.
- `results/experiments/` contains the full-image validation comparison, the
  per-image scores, and the learning curves and histories for the follow-up runs.
- `results/report/` contains the exact test metrics used in the PDF and the
  metadata linking those results to the checkpoint. The report export and live
  notebook evaluation produce slightly different test metrics.

Set `TRAIN = False` in the notebook to evaluate a saved model. A fresh checkout
loads the bundled composite checkpoint from `results/composite/`. If you have a
separate run under `runs/`, its checkpoint takes precedence. Set `SIGK_RUNS_DIR`
to a fresh directory to evaluate the bundled checkpoint in that case.

## Local Windows environment

From the repository root, launch the prepared virtual environment with PowerShell.

```powershell
.\.venv\Scripts\python.exe -m jupyterlab .\projekt_2\low_light.ipynb
```

Select **Python (SIGK Project 2)** as the notebook kernel and run the cells from
the top. The prepared environment uses CPU PyTorch. Use a CUDA runtime for the
full training run if you have access to one.

To recreate the local environment on another Windows machine, run these commands
from the repository root using Python 3.14.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r projekt_2/requirements.txt jupyterlab ipywidgets
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name sigk-project2 --display-name "Python (SIGK Project 2)"
```

## Google Colab with a GPU

1. Copy `low_light.ipynb`, `models.py`, `RAISE_258.csv`, and `requirements.txt` to
   `MyDrive/SIGK/projekt_2`. Copy `results/composite/` to use the bundled checkpoint
   or `experiments.py` to rerun the follow-up experiments.
2. Select **Runtime → Change runtime type → GPU**.
3. In the first code cell, set the Drive folder in `os.chdir(...)` if you
   used a different location. Run the cell and approve the Drive mount.
4. Run the notebook cells from top to bottom. The device cell should print
   `device: cuda`. The notebook saves data and checkpoints under the Drive folder.

## Runs and learning curves

- Run `loss="composite"` and `loss="original"` with the same settings for the
  report. The default is 50 epochs. Both use the same 70/10/20 split, model seed,
  and per-epoch shuffle and crop seeds.
- Each loss writes to `runs/<loss>/`. Rerun the notebook with the same config to
  resume from `checkpoints/last.pt`. The test cells load `checkpoints/best.pt`,
  selected by validation PSNR. Close another kernel running the same experiment
  before resuming. To start a separate experiment, set `SIGK_RUNS_DIR` to a new
  directory before running the configuration cell.
- `figures/training_curves.png` shows training loss and validation PSNR. When
  both histories are present, the validation PSNR panel shows both runs. The
  evaluation cells write `metrics.csv`, `metrics_per_image.csv`, and predictions
  for the 20 held-out images.

## Follow-up experiments

Use a completed composite training run with `best.pt` and `last.pt` in
`runs/composite/checkpoints/`. Set `RUN_EXPERIMENTS = True` in the final notebook
cell or run this command from the project folder in Colab.

```python
!python -u experiments.py --device cuda
```

The script uses the cached PNG pairs and the config and split from
`runs/composite/checkpoints/last.pt`. It evaluates `best.pt` and `last.pt` on
the same full validation images, then trains the original loss and the
composite loss with `lr=3e-4`. Both variants start from the reference seed and
keep its epoch budget. Each variant selects its checkpoint using the same
center-crop PSNR as the reference. Full-image PSNR, SSIM, and LPIPS are then
computed for that checkpoint.

Pass `--step validate`, `--step original`, or `--step lower-lr` to run just one
stage. The new runs resume from their own checkpoints when rerun. Results go
to `runs/experiments/` by default, including `validation_summary.csv`,
per-image CSVs, and `validation_psnr.png`. Use a fresh `--output-dir` if you
switch to a different `--composite-run`. The 20 test images stay reserved for
the final test.
