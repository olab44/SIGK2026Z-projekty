# SIGK Project 3 – neural Phong renderer (conditional GAN)

```bash
pip install -r requirements.txt
cd src
python main.py --window headless     # 3000 images -> ../data/phong/
python ../scripts/check_dataset.py
python ../scripts/make_split.py
python -m neural.train --smoke       # quick check; then: python -m neural.train --run_name relative_gan
python main.py neural --checkpoint ../runs/relative_gan/best.pt   # live comparison (SPACE = new scene)
```

Ablations: `--no_gan`, `--encoding raw`, `--pixel_rel_maps`.

## Kaggle

The dataset is committed in git (`data/phong`, 12 MB);

```bash
git add -A && git commit -m "data + job" && git push
uvx --with six kaggle kernels push -p jobs/2026-11-24_01_gan-v1        # set <kaggle-user> in kernel-metadata.json first
uvx --with six kaggle kernels status <kaggle-user>/sigk3-gan-v1
```

Fetch the results (waits until the kernel finishes, then downloads `best.pt`, `losses.csv`, samples into `runs/`),
then evaluate on the 600 test images (FLIP, LPIPS, SSIM, Hausdorff + figures):

```bash
python scripts/fetch_kaggle.py <kaggle-user>/sigk3-gan-v1
cd src && python -m neural.evaluate --checkpoint ../runs/relative_gan/best.pt
# -> ../runs/relative_gan/eval_test/{results.md, results.json, per_image.csv, images/, figures/}
```
