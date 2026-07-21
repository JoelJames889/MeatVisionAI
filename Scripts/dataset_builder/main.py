from scanner import DatasetScanner
from copier import DatasetCopier

scanner = DatasetScanner()

datasets = scanner.scan()

copier = DatasetCopier()

copier.copy_dataset(datasets)
