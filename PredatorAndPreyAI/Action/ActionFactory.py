from Action.Create import Create
from Action.Evaluate import Evaluate
from Action.Train import Train

FIRST_LEVEL_ID = '1-1'


class ActionFactory:
    def __init__(self, arguments):
        self.arguments = arguments
        self.operation = arguments[1]
        self.path = arguments[2]
        self.nn_id = arguments[3] if len(arguments) > 3 else None

    def create(self):
        if self.operation == 'train':
            return Train(self.path)

        elif self.operation == 'evaluate':
            return Evaluate(self.path)

        elif self.operation == 'create':
            return Create(self.path, self.nn_id)

        else:
            raise Exception("Wrong Command")
