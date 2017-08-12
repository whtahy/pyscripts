import numpy

from pandas import unique


# http://stackoverflow.com/a/11144716
def cartesian(x, y):
    x = pandas.unique(x)
    y = pandas.unique(y)
    return numpy.transpose([numpy.tile(x, len(y)),
                            numpy.repeat(y, len(x))])


def seq(start, end, step=1, dtype=None, exclude=[]):
    if dtype is None and isinstance(step, int):
        dtype = 'int'
    arr = numpy.arange(start=start, stop=end, step=step, dtype=dtype)
    endpoint = (arr[-1] + step).astype(dtype)
    if endpoint <= end:
        arr = numpy.append(arr, endpoint)
    return arr[~numpy.in1d(arr, exclude)]


def numpy_info(np_array, col_width=40):
    otype = str(type(np_array))
    dtype = np_array.dtype.name
    memory = numpy.floor(np_array.data.nbytes / 1000000)
    size = np_array.size
    dims = np_array.shape
    rows = dims[0]
    if len(dims) > 1:
        cols = np_array.shape[1]
    else:
        cols = 1
    shape = f'{rows} x {cols}'
    print(f'Dtype  {dtype:>{col_width}}')
    print(f'Size   {size:>{col_width}}')
    print(f'Shape  {shape:>{col_width}}')
    print(f'Type   {otype:>{col_width}}')
    print(f'Memory {memory:>{col_width-3}} MB')