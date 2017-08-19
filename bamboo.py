import pandas
import numpy
import math


def count_dtypes(df):
    return df.dtypes.value_counts(ascending=True)


def idx_by_feature(df):
    return dict(zip(df.columns, numpy.arange(len(df.columns))))


def print_preview(df, nrows=3, cols=slice(None)):
    dtype_counts = count_dtypes(df)
    print(f'{nrows} first rows')
    print()

    for dtype in dtype_counts.index:
        print(f'{dtype.name}: {dtype_counts[dtype]}')
        print(df.select_dtypes(include=[dtype]).iloc[0:nrows, cols])
        print()


def peek(file_path,
         ext='csv',
         usecols=None,
         show_dtypes=False,
         show_preview=False,
         nrows=100,
         float_bits=32,
         nrows_preview=3,
         header='infer'):
    # Check file ext
    if file_path.endswith('.csv'):
        path = file_path
    else:
        path = f'{file_path}.{ext}'
    df = pandas.read_csv(path, header=header, nrows=nrows, usecols=usecols)

    # Convert to float32
    for col in df.select_dtypes(include=['float']):
        df[col] = df[col].astype(f'float{float_bits}')

    features_by_dtype(df, print_out=show_dtypes)

    if show_preview:
        print_preview(df)

    return df


def features_by_dtype(df,
                      print_out=False,
                      max_per_row=10,
                      min_rows=1,
                      col_padding=2,
                      terminal_width=150):
    features_dict = dict()
    dtype_counts = count_dtypes(df)
    if print_out:
        print(f'{sum(dtype_counts)} features')
        print(f'{len(df.index)} rows, {df.isnull().values.sum().sum()} missing values')
        print()
    for dtype in [x for x in dtype_counts.index]:
        features = df.select_dtypes(include=[dtype]).columns.values
        features_dict[dtype.name] = list(features)
        if print_out:
            idx = idx_by_feature(df)
            length_index = len(str(len(features)))
            length_cols = max([len(str(x)) for x in features]) + col_padding
            names_per_row = min(math.floor(terminal_width / (length_cols + length_index)), max_per_row)
            print(f'{dtype.name}: {dtype_counts[dtype]}')
            col_index = 0
            for feature in features:
                col_index += 1
                print(f'{idx[feature]:>{length_index}} {feature:<{length_cols}}', end='')
                if col_index % names_per_row == 0 \
                        or len(features) < min_rows * names_per_row \
                        or col_index >= len(features):
                    print()
            print()
    return features_dict