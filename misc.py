


def strip_ext(s, ext, sep='.'):
    if ext[0] == sep:
        ext = ext[1:]
    if s.endswith(f'{sep}{ext}'):
        return s[:-len(ext)-len(sep)]
    else:
        return s


def add(*tuples):
    return [sum(x) for x in zip(*tuples)]


def compare(*objects, index=0, print_results=True):
    target = objects[index]
    results = [None] * len(objects)
    for i in range(0, len(objects)):
        results[i] = objects[i] is target
        if print_results:
            print(f"{results[i]}\t {loc_of(objects[i])}\t {objects[i]}")
    return results


def loc_of(*objects):
    locs = [id(obj) for obj in objects]
    if len(locs) == 1:
        return locs[0]
    else:
        return locs