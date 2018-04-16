import time


def timeit(method):
    """
    Decorator to time the execution of the decorated method/function.
    """
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f'{method.__name__}: {(te - ts)/60:.2f} mins, '
              f'{(te - ts):.2f} seconds')
        return result
    return timed
