from Action.ActionFactory import ActionFactory
import sys

action_factory = ActionFactory(sys.argv)
action = action_factory.create()
action.execute()
