from config import Config
from models import RunnerFactory

import tyro


def main(cfg: Config):
    runner = RunnerFactory.start(cfg)
    runner.run()
    runner.make_report()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
