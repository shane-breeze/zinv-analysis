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

        for cat, subdict in self.attributes.items():
            for attr, subsplit in subdict.items():
                if not isinstance(subsplit, dict):
                    self.attributes[cat][attr] = {
                        "evattrs": [{"label": "", "source": "", "nsig": 0.}],
                        "selection": subsplit,
                    }

    def begin(self, event):
        self.database_path = "/".join(os.path.abspath("result.db").split("/")[-3:])
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
            subdict["selection"]: Lambda(subdict["selection"])
            for _, subdict in self.attributes.items()
        }

    def event(self, event):
        data, keys = {}, []
        for attr, subdict in self.attributes.items():
            for var_dict in subdict["evattrs"]:
                val = self.lambda_functions[subdict["selection"]](
                    event, var_dict["source"], var_dict["nsig"],
                )

                label = (
                    "_".join([attr, var_dict["label"]])
                    if var_dict["label"] != "" else
                    attr
                )
                data[label] = val
                keys.append(label)

        df = (
            pd.DataFrame(data, columns=keys)
            .to_sql(self.name, self.engine, if_exists='append', chunksize=1000)
        )

        # Just incase it hangs around in memory for too long
        del df

        for source, nsig in self.variations:
            data, keys = {}, []
            for attr, subdict in self.attributes.items():
                for var_dict in subdict["evattrs"]:
                    val = self.lambda_functions[subdict["selection"]](
                        event, source, nsig,
                    )

                    label = (
                        "_".join([attr, var_dict["label"]])
                        if var_dict["label"] != "" else
                        attr
                    )
                    data[label] = val
                    keys.append(label)

            updown = "Up" if nsig>=0. else "Down"
            table_name = (
                "_".join([self.name, source+updown])
                if source != "" else
                self.name
            )
            df = (
                pd.DataFrame(data, columns=keys)
                .to_sql(table_name, self.engine, if_exists='append', chunksize=1000)
            )
            del df

    def merge(self, other):
        #engine = create_engine("sqlite:///{}".format(os.path.abspath(self.database_path)))
        #connection = engine.connect()
        #connection.execute("""ATTACH DATABASE '{}' AS DB2;""".format(os.path.abspath(other.database_path)))
        #connection.execute("""INSERT INTO {} SELECT * FROM DB2.{};""".format(self.name, other.name))
        #connection.close()
        return self

    def end(self):
        self.lambda_functions = None
        self.engine = None
