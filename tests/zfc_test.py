# Released under CC0.
# Summary: https://creativecommons.org/publicdomain/zero/1.0/
# Legal Code: https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt


import numpy as np

from pyscripts.zfc import (
    cartesian, interleave, is_float, is_int, n_dec, np_hstack, np_info,
    np_ncols, np_nrows, np_vstack, pairtable, seq, triangle_n, triangle_n_inv
)


def test_cartesian():
    assert np.array_equal(
            cartesian(np.array([1, 2, 3]),
                      np.array([4, 5, 6])),
            np.array([
                [1, 4],
                [1, 5],
                [1, 6],
                [2, 4],
                [2, 5],
                [2, 6],
                [3, 4],
                [3, 5],
                [3, 6]
            ])
    )


def test_interleave():
    assert interleave(list('abc'), list('def')) == \
           ['a', 'd', 'b', 'e', 'c', 'f']
    assert interleave(list('abc'), list('defz')) == \
           ['a', 'd', 'b', 'e', 'c', 'f', 'z']
    assert interleave(list('abcz'), list('def')) == \
           ['a', 'd', 'b', 'e', 'c', 'f', 'z']


def test_is_float():
    assert is_float(0.)
    assert is_float(10.)
    assert is_float(0.123)
    assert is_float(10.123)
    assert is_float(np.float32(0))
    assert is_float(np.float64(0))

    assert not is_float(0)
    assert not is_float(123)
    assert not is_float(np.int32(0))
    assert not is_float(np.int64(123))


def test_is_int():
    assert is_int(0)
    assert is_int(123)
    assert is_int(np.int32(0))
    assert is_int(np.int64(123))

    assert not is_int(0.)
    assert not is_int(10.)
    assert not is_int(0.123)
    assert not is_int(10.123)
    assert not is_int(np.float32(0))
    assert not is_int(np.float64(0))


def test_np_hstack():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    assert np.array_equal(np_hstack(a, b, c), np.hstack((a, b, c)))


def test_np_info():
    d = np_info(seq(1, 100).reshape(10, 10))

    assert d['type'] == np.ndarray
    assert d['dtype'] == np.int32
    assert d['size'] == 100
    assert d['shape'] == (10, 10)
    assert d['nbytes'] == 400

    assert len(d.keys()) == 5


def test_np_ncols():
    assert np_ncols(np.array(
            5
    )) is None

    assert np_ncols(np.array(
            [1, 2, 3, 4, 5]
    )) == 5

    assert np_ncols(np.array(
            [[1, 2, 3, 4, 5],
             [2, 3, 4, 5, 6]]
    )) == 5

    assert np_ncols(np.array(
            [[1],
             [2],
             [3],
             [4],
             [5]]
    )) == 1


def test_np_nrows():
    assert np_nrows(np.array(
            5
    )) is None

    assert np_nrows(np.array(
            [1, 2, 3, 4, 5]
    )) == 1

    assert np_nrows(np.array(
            [[1, 2, 3, 4, 5],
             [2, 3, 4, 5, 6]]
    )) == 2

    assert np_nrows(np.array(
            [[1],
             [2],
             [3],
             [4],
             [5]]
    )) == 5


def test_np_vstack():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    assert np.array_equal(np_vstack(a, b, c), np.vstack((a, b, c)))


def test_pairtable():
    assert np.array_equal(
            pairtable(np.array([1, 2, 3, 4, 5, 6])),
            np.array([
                [1, 2, 3],
                [None, 4, 5],
                [None, None, 6]
            ])
    )


def test_triangle_n():
    assert triangle_n(0) == 0
    assert triangle_n(1) == 1
    assert triangle_n(2) == 3
    assert triangle_n(3) == 6
    assert triangle_n(4) == 10

    assert triangle_n(5) == 15
    assert triangle_n(100) == 5050


def test_triangle_n_inv():
    assert triangle_n_inv(0) == 0
    assert triangle_n_inv(1) == 1
    assert triangle_n_inv(2) is None
    assert triangle_n_inv(3) == 2
    assert triangle_n_inv(4) is None
    assert triangle_n_inv(5) is None
    assert triangle_n_inv(6) == 3
    assert triangle_n_inv(7) is None
    assert triangle_n_inv(8) is None
    assert triangle_n_inv(9) is None
    assert triangle_n_inv(10) == 4

    assert triangle_n_inv(15) == 5
    assert triangle_n_inv(5050) == 100


def test_n_dec():
    assert n_dec(100) == 0
    assert n_dec(10) == 0
    assert n_dec(0) == 0
    assert n_dec(0.) == 0
    assert n_dec(0.1) == 1
    assert n_dec(0.12) == 2
    assert n_dec(0.123) == 3
    assert n_dec(0.1234) == 4
    assert n_dec(0.12345) == 5


def test_seq():
    # float
    assert np.array_equal(
            seq(0, 1, 0.1),
            np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    )
    assert np.array_equal(
            seq(1, 0, -0.1),
            np.array([1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.])
    )

    assert np.array_equal(seq(1, 10, 2.5), np.array([1., 3.5, 6., 8.5]))
    assert np.array_equal(seq(10, 1, -2.5), np.array([10, 7.5, 5., 2.5]))

    # int
    assert np.array_equal(seq(1, 5), np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(seq(5, 1), np.array([5, 4, 3, 2, 1]))

    assert np.array_equal(seq(-1, -5), np.array([-1, -2, -3, -4, -5]))
    assert np.array_equal(seq(-5, -1), np.array([-5, -4, -3, -2, -1]))

    assert np.array_equal(seq(1, 5.1), np.array([1, 2, 3, 4, 5]))
    assert np.array_equal(seq(1, 4.9), np.array([1, 2, 3, 4]))
    assert np.array_equal(seq(5, 1.1), np.array([5, 4, 3, 2]))

    assert np.array_equal(seq(-1, -5.1), np.array([-1, -2, -3, -4, -5]))
    assert np.array_equal(seq(-1, -4.9), np.array([-1, -2, -3, -4]))
    assert np.array_equal(seq(-5, -1.1), np.array([-5, -4, -3, -2]))

    assert np.array_equal(seq(1, 10, 2), np.array([1, 3, 5, 7, 9]))
    assert np.array_equal(seq(10, 1, -2), np.array([10, 8, 6, 4, 2]))
