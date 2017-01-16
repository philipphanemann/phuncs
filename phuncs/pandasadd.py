from itertools import islice
from numpy import array


def sliding_window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable

    s -> (s0,s1,...s[n]), (s1,s2,...,s[n+1])
    see: daniel-dipaolo (stackoverflow) '[...]sliding-window-iterator-in-python'
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def series_to_batches(series, step_length):
    """ return 2D!! array of subarrays with step_length [batches x step_length]

    series: pandas series, numpy array or iterable
    step_lengt: size of each sub series
    """
    return array(list(sliding_window(series, step_length)))
