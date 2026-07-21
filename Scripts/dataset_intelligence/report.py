import pandas as pd


class Report:

    def save(self, rows, path):

        df = pd.DataFrame(rows)

        df.to_csv(path, index=False)

        print(df)
