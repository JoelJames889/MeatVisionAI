# ⚡ MeatVision AI — High-Speed Google Colab GPU Training

This directory contains Google Colab GPU notebooks to train both **Species Identification** and **Freshness Detection** models in **~2 to 3 minutes** using free T4 GPU acceleration.

---

### 🚀 Step-by-Step Instructions

1. **Open Google Colab**:
   * Navigate to [colab.research.google.com](https://colab.research.google.com/).

2. **Upload Notebook**:
   * Click **Upload** and select [`notebooks/train_meatvision_colab.ipynb`](train_meatvision_colab.ipynb).

3. **Enable GPU**:
   * In Colab, click **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** $\rightarrow$ **Save**.

4. **Upload Dataset / Project**:
   * Option A: Upload `MeatVisionAI.zip` directly into Colab runtime.
   * Option B: Mount Google Drive if your `Dataset/` folder is stored in Google Drive.

5. **Run Training**:
   * Run all cells sequentially. Training takes $\sim 2-3$ minutes for both models.

6. **Download Checkpoints**:
   * The notebook will prompt you to download `species_model.pth` and `freshness_model.pth`.
   * Place the downloaded `.pth` files into your local project's [`models/`](../models) directory.
   * Restart your local server (`uvicorn backend.app:app`) and enjoy 95%+ accurate live predictions!
