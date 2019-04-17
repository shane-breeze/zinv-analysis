import awkward as awk

from cachetools import LRUCache
from zinv.utils.cache_funcs import get_size

class EventTools(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def begin(self, event):
        event.nsig = 0
        event.source = ''

        # Not callable but want it to persist on across event blocks
        # Return object sizes in bytes
        event._nonbranch_cache["cache"] = LRUCache(self.maxsize, get_size)

    def event(self, event):
        event.MET_pt
        event.nsig = 0
        event.source = ''

        # Remove anything from the previous event block
        event.cache.clear()
