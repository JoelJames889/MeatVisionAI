from .species_builder import ensure_species_folders
from .freshness_builder import ensure_freshness_folders


def run_pipeline():
    ensure_species_folders()
    ensure_freshness_folders()
    print("Dataset pipeline folders created.")


if __name__ == "__main__":
    run_pipeline()
