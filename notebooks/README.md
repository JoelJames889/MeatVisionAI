# Colab GPU training (Species model)

This folder contains a Colab notebook for training `Scripts/models/train_species.py` using the free GPU.

## 1) Put the dataset in Google Drive (recommended option)
Because the full project is ~43GB, this setup assumes you upload/copy **only the dataset** to Drive (and not the whole project).

Expected Drive paths inside MyDrive:
- `MyDrive/Dataset/train/species`
- `MyDrive/Dataset/valid/species`
- `MyDrive/Dataset/test/species`

The notebook will:
- Mount Drive
- Clone/copy the project code into Colab’s `/content` (small)
- Copy/symlink the dataset from Drive into `/content/Dataset` so the existing training code can run unchanged.

## 2) Open the notebook
Open:
- `train_species_gpu.ipynb`

Then run cells top-to-bottom.

## 3) What gets produced
After training, you should see:
- `Models/species/species_model.pth` under the code directory in Colab

## Notes
- Training code chooses `cuda` automatically when available.
- If you see import errors (`from config import *`), ensure the notebook adds `Scripts/models` to `sys.path` (it does).

