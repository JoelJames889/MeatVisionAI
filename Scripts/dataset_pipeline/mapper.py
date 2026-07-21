from config import SPECIES_KEYWORDS, FRESHNESS_KEYWORDS


class DatasetMapper:

    def __init__(self):
        pass

    def get_species(self, text):

        text = text.lower()

        for species, keywords in SPECIES_KEYWORDS.items():

            for keyword in keywords:

                if keyword in text:
                    return species

        return None

    def get_freshness(self, text):

        text = text.lower()

        # Fresh must come before Half Fresh
        if any(x in text for x in ["fresh", "meat_fresh", "poultry_fresh"]):

            if "half" not in text:
                return "fresh"

        if any(x in text for x in ["half-fresh", "half_fresh", "half fresh"]):
            return "half_fresh"

        if any(
            x in text
            for x in ["spoiled", "rotten", "stale", "meat_spoiled", "poultry_spoiled"]
        ):
            return "spoiled"

        # Beef Freshness Dataset (A1-A5)

        if "a1" in text:
            return "fresh"

        if "a2" in text:
            return "half_fresh"

        if "a3" in text:
            return "spoiled"

        if "a4" in text:
            return "spoiled"

        if "a5" in text:
            return "spoiled"

        return None

    def classify(self, path):

        text = str(path).replace("\\", "/").lower()

        species = self.get_species(text)

        freshness = self.get_freshness(text)

        return species, freshness
