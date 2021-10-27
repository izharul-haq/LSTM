class InputError(Exception):
    def __init__(self, message):
        self.__message = message
        super().__init__(message)

    def __str__(self):
        return self.__message
