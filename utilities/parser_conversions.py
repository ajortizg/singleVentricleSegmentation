
def str_to_tuple(x, dtype=float):
    return tuple(map(dtype, x.strip().replace('(', '').replace(')', '').split(',')))
