import pandas as pd
import yaml

from zinv.utils.Lambda import Lambda

class ParquetReader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        with open(self.cfg, 'r') as f:
            cfg = yaml.load(f)

        self.name = cfg["name"]
        self.attributes = cfg["attributes"]
        self.variations = cfg["variations"]
        self.dtypes = cfg["dtypes"]

    def begin(self, event):
        data_or_mc = "Data" if event.config.dataset.isdata else "MC"
        attributes = self.attributes["Both"]
        attributes.update(self.attributes[data_or_mc])
        self.attributes = attributes

        self.lambda_functions = {
            selection: Lambda(selection)
            for _, selection in self.attributes.items()
        }

    def end(self):
        self.lambda_functions = None

    def event(self, event):
        opts = ('', 0.)
        path = "{}.pq".format(self.name)
        for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
            df.to_parquet(path, engine='pyarrow', compression='snappy')
        print("Created {}".format(path))

        for source, nsig in self.variations:
            opts = (source, nsig)
            updown = "Up" if nsig>=0. else "Down"
            name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )
            path = "{}.pq".format(name)

            for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
                df.to_parquet(path, engine='pyarrow', compression='snappy')
            print("Create {}".format(path))

    def chunk_events(self, event, opts=[], chunksize=int(1e5)):
        #for start in xrange(0, event.size, chunksize):
            #stop = start+chunksize if start+chunksize<event.size else event.size

        #data = {
        #    attr: self.lambda_functions[selection](event, *opts)[start:stop]
        data = {
            attr: self.lambda_functions[selection](event, *opts)
            for attr, selection in self.attributes.items()
        }

        yield (
            pd.DataFrame(data, columns=self.attributes.keys())
            .astype({k: self.dtypes[k] for k in self.attributes.keys()})
        )
