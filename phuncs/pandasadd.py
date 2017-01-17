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


def series_to_input_batches(series, step_length):
    """ return 2D!! array of subarrays with step_length [batches x step_length]

    series: pandas series, numpy array or iterable
    step_lengt: size of each sub series
    """
    return array(list(sliding_window(series, step_length)))


def series_to_output_batches(series, step_length, out_length=1):
    """ return 2D!! array of subarray with predicted_length [batches x predicted length]

    series: pandas series, numpy array or iterable
    step_length: data that is given/not predicted
    pred_length: prediction length. Default is on step foreward
    """
    all_vals = series_to_input_batches(series, out_length)
    return all_vals[step_length:]


