import os
import shutil
import time
import numpy as np
import pandas as pd
import yaml
from scipy.special import erfinv
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
            list(self.attribute_timing.keys())
            + list(other.attribute_timing.keys())
        ):
            attribute_timing[keys] = (
                self.attribute_timing.get(keys, 0.)
                + other.attribute_timing.get(keys, 0.)
            )
        self.attribute_timing = attribute_timing

    def begin(self, event):
        with open(self.hdf5_config_path, 'r') as f:
            cfg = yaml.load(f)
        self.name = cfg["name"]
        self.attributes = cfg["attributes"]
        self.variations = cfg["variations"]
        self.dtypes = cfg["dtypes"]

        task = os.path.basename(os.getcwd())
        outfile = "result.h5"
        if "SGE_TASK_ID" in os.environ:
            outfile = "result_{:05d}.h5".format(int(os.environ["SGE_TASK_ID"])-1)
        self.path = os.path.join(self.outdir, outfile)

        if os.path.exists(self.path):
            print("File already exists {}".format(self.path))

            failed_dir = os.path.join(os.path.dirname(self.path), "failed")
            fail_idx = 0
            failed_path = os.path.join(
                os.path.dirname(self.path),
                "failed",
                os.path.basename(self.path),
            ) + ".{:05d}".format(fail_idx)
            while os.path.exists(failed_path):
                fail_idx += 1
                failed_path = failed_path[:-5] + "{:05d}".format(fail_idx)

            shutil.move(self.path, failed_path)

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
        if len(self.variations) == 0:
            for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
                df.to_hdf(
                    self.path, self.name, format='table', append=True,
                    complevel=9, complib='zlib',
                )

        for source, vlabel, vval in self.variations:
            if vlabel == "percentile":
                nsig = np.sqrt(2)*erfinv(2*vval/100.-1)
                table_name = (
                    "_".join([self.name, "{}{}".format(source, vval)])
                    if source != "" else
                    self.name
                )
            elif vlabel == "sigmaval":
                nsig = vval
                updown = "Up" if nsig>=0. else "Down"
                updown = "{:.2f}".format(np.abs(nsig)).replace(".", "p") + updown
                table_name = (
                    "_".join([self.name, "{}{}".format(source, updown)])
                    if source != "" else
                    self.name
                )
            elif vlabel.lower() in ["up", "down"]:
                nsig = 1. if vlabel.lower()=="up" else -1.
                table_name = (
                    "_".join([self.name, "{}{}".format(source, vlabel)])
                    if source != "" else
                    self.name
                )
            else:
                nsig = 0.
                table_name = self.name

            opts = (source, nsig)
            for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
                df.to_hdf(
                    self.path, table_name, format='table', append=True,
                    complevel=9, complib='zlib',
                )

    def chunk_events(self, event, opts=[], chunksize=int(1e7)):
        # currently not chunking
        data = {}

        pbar = tqdm(self.attributes.items(), unit='attr')
        for attr, selection in pbar:
            pbar.set_description("{}, {}, {:.2f}".format(attr, *opts))
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

        yield (
            pd.DataFrame(data, columns=self.attributes.keys())
            .astype({k: self.dtypes[k] for k in self.attributes.keys()})
        )
