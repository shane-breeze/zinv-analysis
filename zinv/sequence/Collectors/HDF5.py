import os
import numpy as np
import pandas as pd
import yaml
import tqdm

from zinv.utils.Lambda import Lambda

class HDF5Reader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.hdf5_config_path, 'r') as f:
            cfg = yaml.load(f)
        self.name = cfg["name"]
        self.attributes = cfg["attributes"]
        self.variations = cfg["variations"]
        self.dtypes = cfg["dtypes"]

        task = os.path.basename(os.getcwd())
        self.path = os.path.join(
            self.outdir, task.replace("task", "result")+".h5",
        )

        try:
            pd.DataFrame().to_hdf(
                self.path, self.name, mode='w', format='table', complevel=9,
                complib='blosc:lz4hc',
            )
        except IOError:
            pass
        for source, nsig in self.variations:
            updown = "Up" if nsig>=0. else "Down"
            #updown = "{:.2f}".format(np.abs(nsig)).replace(".", "p") + updown
            table_name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )
            try:
                pd.DataFrame().to_hdf(
                    self.path, table_name, mode='w', format='table',
                    complevel=9, complib='blosc:lz4hc',
                )
            except IOError:
                pass

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
        for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
            df.to_hdf(
                self.path, self.name, format='table', append=True,
                complevel=9, complib='blosc:lz4hc',
            )
        print("Created result.h5 with table {}".format(self.name))

        for source, nsig in self.variations:
            opts = (source, nsig)
            updown = "Up" if nsig>=0. else "Down"
            #updown = "{:.2f}".format(np.abs(nsig)).replace(".", "p") + updown
            table_name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )

            for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
                df.to_hdf(
                    self.path, table_name, format='table', append=True,
                    complevel=9, complib='blosc:lz4hc',
                )
                print("Create result.h5 with table {}".format(table_name))

    def chunk_events(self, event, opts=[], chunksize=int(1e5)):
        # currently not chunking
        data = {
            attr: self.lambda_functions[selection](event, *opts)
            for attr, selection in tqdm.tqdm(self.attributes.items(), unit='attr', dynamic_ncols=True)
        }

        yield (
            pd.DataFrame(data, columns=self.attributes.keys())
            .astype({k: self.dtypes[k] for k in self.attributes.keys()})
        )
