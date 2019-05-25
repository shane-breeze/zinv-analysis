import os
import numpy as np
import pandas as pd
import yaml

from sqlalchemy import create_engine

from zinv.utils.Lambda import Lambda

class SqliteReader(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        with open(self.cfg, 'r') as f:
            cfg = yaml.load(f)

        self.name = cfg["name"]
        self.attributes = cfg["attributes"]
        self.variations = cfg["variations"]

    def begin(self, event):
        self.engine = create_engine('sqlite:///result.db')
        conn = self.engine.connect()

        for source, nsig in [["", 0.]]+self.variations:
            updown = "Up" if nsig>0. else "Down" if nsig<0. else ""

            table_name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )

            conn.execute("DROP TABLE IF EXISTS {};".format(table_name))

        conn.close()

        data_or_mc = "Data" if event.config.dataset.isdata else "MC"
        attributes = self.attributes["Both"]
        attributes.update(self.attributes[data_or_mc])
        self.attributes = attributes

        self.lambda_functions = {
            selection: Lambda(selection)
            for _, selection in self.attributes.items()
        }

    def event(self, event):
        opts = ('', 0.)
        for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
            df.to_sql(self.name, self.engine, if_exists='append', chunksize=100000)
        del df
        print("Created result.db with table {}".format(self.name))

        for source, nsig in self.variations:
            opts = (source, nsig)
            updown = "Up" if nsig>=0. else "Down"
            table_name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )

            for df in self.chunk_events(event, opts=opts, chunksize=int(1e7)):
                df.to_sql(table_name, self.engine, if_exists='append', chunksize=100000)
            del df
            print("Create result.db with table {}".format(table_name))

    def chunk_events(self, event, opts=[], chunksize=int(1e5)):
        for start in xrange(0, event.size, chunksize):
            stop = start+chunksize if start+chunksize<event.size else event.size

            data = {
                attr: self.lambda_functions[selection](event, *opts)[start:stop]
                for attr, selection in self.attributes.items()
            }

            yield pd.DataFrame(data, columns=self.attributes.keys())

    def end(self):
        self.lambda_functions = None
        self.engine = None
