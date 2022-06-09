import argparse


class ArgParser:
    def __init__(self):
        self.args = ArgParser.initialize_parser()

    @staticmethod
    def initialize_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--forward-num-epochs",
            "--fe",
            type=int,
            default=None,
            help="Number of epochs for forward model",
        )
        parser.add_argument(
            "--num-wavelens",
            type=int,
            default=800,
            help="Number of wavelens to interpolate to",
        )
        parser.add_argument(
            "--backward-num-epochs",
            "--be",
            type=int,
            default=None,
            help="Number of epochs for backward model",
        )
        parser.add_argument(
            "--forward-batch-size",
            "--fbs",
            type=int,
            default=None,
            help="Batch size for forward model",
        )
        parser.add_argument(
            "--num-samples",
            type=int,
            default=1,
            help="How many runs for optimization",
        )
        parser.add_argument(
            "--backward-batch-size",
            "--bbs",
            type=int,
            default=None,
            help="Batch size for backward model",
        )
        parser.add_argument(
            "--prediction-iters",
            type=int,
            default=1,
            help="Number of iterations to run predictions",
        )
        parser.add_argument(
            "--use-cache",
            type=eval,
            choices=[True, False],
            default=False,
            help="Load saved dataset (avoids 1 minute startup cost of fetching data from database, useful for quick tests).",
        )
        parser.add_argument(
            "--use-forward",
            type=eval,
            choices=[True, False],
            default=True,
            help="Whether to use a forward model at all",
        )
        parser.add_argument(
            "--load-forward-checkpoint",
            type=eval,
            choices=[True, False],
            default=False,
            help="Load trained forward model. Useful for validation. Requires model to already be trained and saved.",
        )
        parser.add_argument(
            "--load-backward-checkpoint",
            type=eval,
            choices=[True, False],
            default=False,
            help="Load trained backward model. Useful for validation. Requires model to already be trained and saved.",
        )
        return parser.parse_args()
