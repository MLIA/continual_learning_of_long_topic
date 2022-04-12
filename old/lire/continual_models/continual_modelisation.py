class ContinualModel(object):

    def __init__(self):
        super().__init__()

    def at_end_train(self, **kwargs) -> None:
        raise NotImplementedError
