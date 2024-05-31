from argparse import Namespace


def parse_cfg_args(path) -> Namespace:
    with open(path, "r") as f:
        cfg_args = f.read()
    return eval(cfg_args)
