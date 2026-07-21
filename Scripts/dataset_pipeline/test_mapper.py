from mapper import DatasetMapper

mapper = DatasetMapper()

tests = [
    "train/Fresh",
    "train/Half-Fresh",
    "train/Spoiled",
    "train/A1",
    "train/A2",
    "train/A3",
    "beef/train/images",
    "Detect Pork/train/images",
    "Chicken Meat/train/images",
    "Fish/Fresh",
]

for t in tests:

    print(t, "------>", mapper.classify(t))
