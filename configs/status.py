class StatusClass:
    """
    Global variables mutable during training
    """

    def __init__(self):
        self.time = 0
        self.cache = dict()


Status = StatusClass()
