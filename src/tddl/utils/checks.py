from pathlib import Path


def check_path(path_str):
    "Turn path_str into instance of Path"
    return Path(path_str)


def check_paths(*args):
    "Turn multiple path-strings into instances of Paths"
    return tuple(check_path(path_str) for path_str in args)
