import inspect

def get_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for var_name, var_val in callers_local_vars:
        if var_val is var:
            return var_name 


class ArgLenError(Exception):
    def __init__(self, iterable1, iterable2):
        iterable1_name = get_name(iterable1)
        iterable2_name = get_name(iterable2)
        super().__init__("Iterable {} is of length {} while iterable {} is of length {}".format(iterable1_name, str(len(iterable1)), iterable2_name, str(len(iterable2))))

class NoHeaderError(Exception):
    def __init__(self, file):
        super().__init__("The file {} doesn't have a properly formatted header".format(file.name))