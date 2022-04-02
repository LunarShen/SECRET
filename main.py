from __future__ import print_function, absolute_import
from secret.config import get_cfg
from secret.utils.defaults import default_argument_parser, default_setup
from secret.engine import create_engine

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    engine = create_engine(cfg)
    engine.run()
    return

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
