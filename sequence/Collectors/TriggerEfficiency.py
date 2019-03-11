from .Histogrammer import HistReader, HistCollector

TriggerEfficiencyReader = HistReader

class TriggerEfficiencyCollector(HistCollector):
    def save(self, histograms):
        df = histograms.histograms

        # Remove variations from name and region
        levels = df.index.names
        df = df.reset_index(["name", "region", "weight"])
        df["name"] = df.apply(lambda row: row["name"].replace(row["weight"], ""), axis=1)
        df["region"] = df.apply(lambda row: row["region"].replace(row["weight"], ""), axis=1)
        df = df.set_index(["name", "region", "weight"], append=True).reorder_levels(levels)

        histograms.histograms = df
        histograms.save(self.outdir)

    def draw(self, histograms):
        return []
