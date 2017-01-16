from phuncs.pandasadd import sliding_window, series_to_batches
import pytest
from collections import namedtuple
from numpy import array, array_equal, atleast_2d, arange
from pandas import Series

Window = namedtuple('Window', ['sequence', 'step_length'])


@pytest.mark.parametrize("test_input,expected", [
    (Window(range(5), 5), [tuple(range(5))]),
    (Window(range(5), 4), [tuple(range(4)), tuple(range(1, 5))]),
    (Window('hello', 5), [tuple('hello')])
])
def test_sliding_window(test_input, expected):
    seq, n = test_input.sequence, test_input.step_length
    assert list(sliding_window(seq, n)) == expected


@pytest.mark.parametrize("test_input,expected", [
    (Window(range(5), 5), atleast_2d(arange(5))),
    (Window(range(5), 4), array((arange(4), arange(1, 5)))),
    (Window(range(5), 3), array(list(arange(*x) for x in
                                     ((0, 3), (1, 4), (2, 5))))),
    (Window(range(5), 2), array(list(arange(*x) for x in
                                     ((0, 2), (1, 3), (2, 4), (3, 5))))),
    (Window(list('hello'), 4), array(list((list('hell'), list('ello')))))
])
def test_series_to_batches(test_input, expected):
    series, n = test_input.sequence, test_input.step_length
    assert array_equal(series_to_batches(Series(series), n), expected)
