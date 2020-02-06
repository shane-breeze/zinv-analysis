import os
import shutil
import time
import numpy as np
import pandas as pd
import awkward as awk
import yaml
from scipy.special import erfinv
from tqdm.auto import tqdm

from zinv.utils.Lambda import Lambda

class HDF5ReaderNew(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        with open(self.hdf5_config_path, 'r') as f:
            cfg = yaml.load(f)
        self.attributes = cfg["tables"]
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

        attributes = {}
        for table_name, subtables in self.attributes.items():
            attributes[table_name] = subtables.get("Both", {})
            attributes[table_name].update(subtables.get(data_or_mc, {}))
        self.attributes = attributes

        self.lambda_functions = {
            selection: Lambda(selection)
            for _, subtable in self.attributes.items()
            for _, selection in subtable.items()
        }

        self.event_size_per_table = {
            table_name: 0 for table_name in self.attributes.keys()
        }
        self.object_size_per_table = {
            table_name: 0 for table_name in self.attributes.keys()
        }

    def end(self):
        self.lambda_functions = None

    def event(self, event):
        # Event table
        for table_name, subtable in self.attributes.items():
            for df in self.chunk_events(
                event, label=table_name, attributes=subtable,
                flatten="Event" not in table_name,
            ):
                #print(df)
                df.to_hdf(
                    self.path, table_name, format='table', append=True,
                    complevel=9, complib='zlib',
                )

    def chunk_events(self, event, label="", attributes={}, flatten=False):
        # currently not chunking
        data = {}

        pbar = tqdm(attributes.items(), unit='attr')
        for attr, selection in pbar:
            pbar.set_description("{} {}".format(label, attr))

            # Process
            attr_val = self.lambda_functions[selection](event)
            data[attr] = attr_val

        if len(data) == 0:
            return [pd.DataFrame()]

        if flatten:
            df = awk.topandas(awk.Table(data), flatten=True).reset_index()
            df.columns = ["parent_event", "object_id"] + list(attributes.keys())
            self.dtypes["parent_event"] = "int32"
            self.dtypes["object_id"] = "int32"
            df.loc[:, "parent_event"] = df["parent_event"] + self.event_size_per_table[label]
        else:
            df = pd.DataFrame(data, columns=attributes.keys())
        df = df.reset_index(drop=True)
        df.index += self.object_size_per_table[label]
        self.event_size_per_table[label] += event.size
        self.object_size_per_table[label] += df.shape[0]

        return [df.astype({k: self.dtypes[k] for k in attributes.keys()})]
