import os
import time
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from zinv.utils.Lambda import Lambda

class HDF5Reader(object):
    def __init__(self, **kwargs):
        self.measure_timing = False
        self.attribute_timing = {}
        self.__dict__.update(kwargs)

    def merge(self, other):
        attribute_timing = {}
        for keys in set(
            self.attribute_timing.keys()+other.attribute_timing.keys()
        ):
            attribute_timing[keys] = (
                self.attribute_timing.get(keys, 0.)
                + other.attribute_timing.get(keys, 0.)
            )
        self.attribute_timing = attribute_timing

#    def collect(self):
#        return self.attribute_timing

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
        #print("Created result.h5 with table {}".format(self.name))

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
                #print("Create result.h5 with table {}".format(table_name))

    def chunk_events(self, event, opts=[], chunksize=int(1e7)):
        # currently not chunking
        data = {}
        for attr, selection in tqdm(self.attributes.items(), unit='attr'):
            # Initialise timing
            if self.measure_timing:
                start = time.time_ns()

            # Process
            attr_val = self.lambda_functions[selection](event, *opts)
            data[attr] = attr_val

            # End timing
            if self.measure_timing:
                end = time.time_ns()
                if attr not in self.attribute_timing:
                    self.attribute_timing[attr] = 0.
                self.attribute_timing[attr] += (end - start)

        data = {
            attr: self.lambda_functions[selection](event, *opts)
            for attr, selection in tqdm(self.attributes.items(), unit='attr')
        }

        yield (
            pd.DataFrame(data, columns=self.attributes.keys())
            .astype({k: self.dtypes[k] for k in self.attributes.keys()})
        )
