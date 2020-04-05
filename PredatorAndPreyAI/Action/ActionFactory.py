from Action.Create import Create
from Action.Evaluate import Evaluate
from Action.Train import Train

FIRST_LEVEL_ID = '1-1'


class ActionFactory:
    def __init__(self, arguments):
        self.arguments = arguments
        self.operation = arguments[1]

    def create(self):
        if self.operation == 'train':
            paths = self.arguments[2:]
            return Train(paths)

        elif self.operation == 'evaluate':
            paths = self.arguments[2:]
            return Evaluate(paths)

        elif self.operation == 'create':
            path = self.arguments[2]
            nn_id = self.arguments[3] if len(self.arguments) > 3 else None
            return Create(path, nn_id)

        else:
            raise Exception("Wrong Command")
