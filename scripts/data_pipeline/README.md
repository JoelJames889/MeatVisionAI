# Data Pipeline Documentation

This directory serves as documentation for the data engineering pipeline that was
executed during the development of MeatVision AI. The pipeline scripts have been
archived since they were one-time utilities that completed their purpose.

## Pipeline Phases

### Phase 1 — Data Extraction
Extracted raw images from multiple source archives (13 datasets, 100K+ images).

### Phase 2 — Deduplication
Used perceptual hashing to identify and remove duplicate images across all sources.

### Phase 3 — Data Cleaning
Scanned every image with PIL to detect and remove corrupted or unreadable files
that would crash the PyTorch training loop.

### Phase 4 — YOLO Label Parsing
Parsed YOLO `.txt` bounding-box annotations for the freshness dataset. Extracted
class integers (0=Fresh, 1=Half-Fresh, 2=Spoiled) and organized images into
structured class folders.

### Phase 5 — Class Balancing & Augmentation
Addressed severe class imbalance using OpenCV and albumentations to dynamically
duplicate and augment minority classes (rotate, flip, brightness adjustment)
until all classes reached approximately **1,118 images** each.

### Phase 5b — AI-Assisted Cleaning
Used a secondary AI model to scan for remaining mislabeled or low-quality images.

### Phase 5c–5e — Web Scraping
Scraped additional Fish and Pork images from Bing to supplement
underrepresented classes.

## Final Dataset Structure

```
Dataset/
├── species/
│   ├── beef/       (~1,118 images)
│   ├── chicken/    (~1,118 images)
│   ├── fish/       (~1,118 images)
│   └── pork/       (~1,118 images)
└── freshness/
    ├── fresh/      (balanced)
    ├── halffresh/  (balanced)
    └── spoiled/    (balanced)
```

## Notes
- The original pipeline scripts are preserved in git history if needed.
- The final cleaned dataset is stored in the `Dataset/` directory (gitignored).
