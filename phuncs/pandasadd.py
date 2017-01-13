from itertools import islice


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
