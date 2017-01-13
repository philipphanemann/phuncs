from phuncs.pandasadd import sliding_window
import pytest
from collections import namedtuple

Window = namedtuple('Window', ['seq', 'n'])


@pytest.mark.parametrize("test_input,expected", [
    (Window(range(5), 5), [tuple(range(5))]),
    (Window(range(5), 4), [tuple(range(4)), tuple(range(1, 5))]),
    (Window('hello', 5), [tuple('hello')])
])
def test_sliding_window(test_input, expected):
    seq, n = test_input.seq, test_input.n
    assert list(sliding_window(seq, n)) == expected
