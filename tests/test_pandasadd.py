from phuncs.pandasadd import (sliding_window, series_to_batches,
                              series_to_batches_predicted)
import pytest
from collections import namedtuple
from numpy import array, array_equal, arange
from pandas import Series

Window = namedtuple('Window', ['sequence', 'step_length'])
Forecast = namedtuple('Forecast', ['sequence', 'step_length', 'pred_length'])


@pytest.mark.parametrize("test_input,expected", [
    (Window(range(5), 5), [tuple(range(5))]),
    (Window(range(5), 4), [tuple(range(4)), tuple(range(1, 5))]),
    (Window('hello', 5), [tuple('hello')])
])
def test_sliding_window(test_input, expected):
    sequence, n = test_input
    assert list(sliding_window(sequence, n)) == expected


@pytest.mark.parametrize("test_input,expected", [
    (Window(range(5), 5), array(arange(5), ndmin=2)),
    (Window(Series(range(5)), 4), array((arange(4), arange(1, 5)))),
    (Window(range(5), 3), array(list(arange(*x) for x in
                                     ((0, 3), (1, 4), (2, 5))))),
    (Window(range(5), 2), array(list(arange(*x) for x in
                                     ((0, 2), (1, 3), (2, 4), (3, 5))))),
    (Window(list('hello'), 4), array(list((list('hell'), list('ello')))))
])
def test_series_to_batches(test_input, expected):
    sequence, n = test_input
    assert array_equal(series_to_batches(sequence, n), expected)


@pytest.mark.parametrize("test_input,expected", [
    (Forecast(range(5), 4, 1), array(4, ndmin=2)),
    (Forecast(range(5), 3, 1), array([[3], [4]])),
    (Forecast(list('hello'), 4, 1), array('o', ndmin=2)),
    (Forecast(Series(range(10)), 7, 2), array([[7, 8], [8, 9]])),
    (Forecast(range(10), 5, 4), array([[5, 6, 7, 8], [6, 7, 8, 9]]))
])
def test_series_to_predicted_values(test_input, expected):
    series, n_step, n_pred = test_input
    calculated = series_to_batches_predicted(series, n_step, n_pred)
    assert array_equal(calculated, expected)

