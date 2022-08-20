from abc import ABC


class Action(ABC):
    @staticmethod
    def is_copy():
        return False

    @staticmethod
    def is_copy_shift():
        return False

    @staticmethod
    def is_deletion():
        return False

    @staticmethod
    def is_substitution():
        return False

    @staticmethod
    def is_insertion():
        return False

    @staticmethod
    def is_noop():
        return False


class Noop(Action):
    @staticmethod
    def is_noop():
        return True


class Copy(Action):
    @staticmethod
    def is_copy():
        return True


class CopyShift(Action):
    @staticmethod
    def is_copy_shift():
        return True


class Deletion(Action):
    @staticmethod
    def is_deletion():
        return True


class Insertion(Action):
    def __init__(self, token: str):
        self._token = token

    @property
    def token(self):
        return self._token

    @staticmethod
    def is_insertion():
        return True


class Substitution(Action):
    def __init__(self, token: str):
        self._token = token

    @property
    def token(self):
        return self._token

    @staticmethod
    def is_substitution():
        return True
