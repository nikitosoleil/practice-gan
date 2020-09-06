class ClassProperty:
    """
    Combines @staticmethod and @property functionality
    """

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, cls):
        return self.f.__get__(obj, cls)()


def classproperty(f):
    f = classmethod(f)
    return ClassProperty(f)
