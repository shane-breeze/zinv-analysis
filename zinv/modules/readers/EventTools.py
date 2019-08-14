import awkward as awk

import operator
import time
from functools import partial
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
from zinv.utils.cache_funcs import get_size

def register_function(event, name, function, self):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, name))
    def cached_function(ev, *args, **kwargs):
        if self.timing:
            start = time.time_ns()
        result = function(ev, *args, **kwargs)
        if self.timing:
            end = time.time_ns()
            if name not in self.function_timing:
                self.function_timing[name] = 0.
            self.function_timing[name] += (end - start)
        return result
    setattr(event, name, cached_function)

class EventTools(object):
    def __init__(self, **kwargs):
        self.timing = True
        self.function_timing = {}
        self.__dict__.update(kwargs)

    def merge(self, other):
        function_timing = {}
        for keys in set(
            self.function_timing.keys()+other.function_timing.keys()
        ):
            function_timing[keys] = (
                self.function_timing.get(keys, 0.)
                + other.function_timing.get(keys, 0.)
            )
        self.function_timing = function_timing

    def collect(self):
        return self.function_timing

    def begin(self, event):
        event.nsig = 0
        event.source = ''

        # Not callable but want it to persist on across event blocks
        # Return object sizes in GB
        event._nonbranch_cache["cache"] = LRUCache(int(self.maxsize*1024**3), get_size)
        event._nonbranch_cache["register_function"] = partial(register_function, self=self)

    def event(self, event):
        event.run
        event.nsig = 0
        event.source = ''

        # Remove anything from the previous event block
        event.cache.clear()
