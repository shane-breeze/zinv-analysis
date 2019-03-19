import logging
logger = logging.getLogger(__name__)

class ScribblerWrapper(object):
    def __init__(self, scribbler):
        logger.debug("{}.__init__".format(scribbler.name))
        self.scribbler = scribbler
        self.data = getattr(self.scribbler, "data", True)
        self.mc = getattr(self.scribbler, "mc", True)

    def __repr__(self):
        return repr(self.scribbler)

    def __getattr__(self, attr):
        if attr in ["scribbler", "data", "mc"]:
            raise AttributeError("{} should be assigned but isn't".format(attr))
        return getattr(self.scribbler, attr)

    def begin(self, event):
        logger.debug("{}.begin".format(self.scribbler.name))
        self.isdata = event.config.dataset.isdata

        if self.isdata and not self.data:
            return True

        if not self.isdata and not self.mc:
            return True

        if hasattr(self.scribbler, "begin"):
            return self.scribbler.begin(event)

    def event(self, event):
        logger.debug("{}.event".format(self.scribbler.name))
        if self.isdata and not self.data:
            return True

        if not self.isdata and not self.mc:
            return True

        if hasattr(self.scribbler, "event"):
            return self.scribbler.event(event)
        return True

    def end(self):
        logger.debug("{}.end".format(self.scribbler.name))
        if hasattr(self.scribbler, "end"):
            return self.scribbler.end()
        return True
