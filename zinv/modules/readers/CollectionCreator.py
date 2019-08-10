class Collection(object):
    """Alias class for collections"""
    def __init__(self, name, event, ref_name=None):
        self.name = name
        self.event = event
        self.ref_name = ref_name

    def __getattr__(self, attr):
        if attr in ["name", "event", "ref_name"]:
            raise AttributeError("{} should be defined but isn't".format(attr))
        return getattr(self.event, self.name+"_"+attr)

    def __repr__(self):
        return "{}(name = {!r}, ref_name = {!r})".format(
            self.__class__.__name__,
            self.name,
            self.ref_name,
        )

class CollectionCreator(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def event(self, event):
        for collection in self.collections:
            setattr(event, collection, Collection(collection, event))
