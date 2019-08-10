import awkward as awk

import operator
from functools import partial
from cachetools import LRUCache, cachedmethod
from cachetools.keys import hashkey
from zinv.utils.cache_funcs import get_size

def register_function(self, name, function):
    @cachedmethod(operator.attrgetter('cache'), key=partial(hashkey, name))
    def cached_function(ev, *args, **kwargs):
        return function(ev, *args, **kwargs)
    setattr(self, name, cached_function)

class EventTools(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.nsig = 0
        event.source = ''

        # Not callable but want it to persist on across event blocks
        # Return object sizes in bytes
        event._nonbranch_cache["cache"] = LRUCache(self.maxsize, get_size)
        event._nonbranch_cache["register_function"] = register_function

    def event(self, event):
        event.run
        event.nsig = 0
        event.source = ''

        # Remove anything from the previous event block
        event.cache.clear()
