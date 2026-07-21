from pathlib import Path


class DatasetResolver:

    def resolve(self, info):

        name = info["dataset"].lower()

        folders = " ".join(info["folders"]).lower()

        text = name + " " + folders

        species = "UNKNOWN"

        freshness = False

        # ---------------- Species ----------------

        if any(x in text for x in ["beef", "cow", "a1", "a2", "a3", "a4", "a5"]):
            species = "BEEF"

        elif any(
            x in text
            for x in [
                "chicken",
                "wing",
                "breast",
                "leg",
                "thigh",
                "drumstick",
                "quater",
            ]
        ):
            species = "CHICKEN"

        elif any(
            x in text
            for x in [
                "fish",
                "trout",
                "shrimp",
                "sea bass",
                "red mullet",
                "horse mackerel",
                "gilt head bream",
                "black sea sprat",
            ]
        ):
            species = "FISH"

        elif any(x in text for x in ["pork", "fork", "horse meat"]):
            species = "PORK"

        # ---------------- Freshness ----------------

        if any(
            x in text for x in ["fresh", "spoiled", "half-fresh", "rotten", "rooten"]
        ):
            freshness = True

        return {"species": species, "freshness": freshness}
