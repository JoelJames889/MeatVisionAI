from config import RAW_DATASET, REPORTS
from analyzer import DatasetAnalyzer
from resolver import DatasetResolver
from report import Report

analyzer = DatasetAnalyzer()

resolver = DatasetResolver()

report = Report()

rows = []

for dataset in RAW_DATASET.iterdir():

    if not dataset.is_dir():
        continue

    info = analyzer.analyze(dataset)

    result = resolver.resolve(info)
    info["species"] = result["species"]
    info["freshness"] = result["freshness"]

    rows.append(info)

report.save(rows, REPORTS / "dataset_report.csv")

print("\nDONE")
