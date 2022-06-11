from torch import Tensor
from torch.nn.functional import mse_loss


class TrainingUtils:
    @staticmethod
    def rmse(pred: Tensor, target: Tensor, epsilon=1e-8):
        """Root mean squared error.

        Epsilon is to avoid NaN gradients. See https://discuss.pytorch.org/t/rmse-loss-function/16540/3.
        """
        return (mse_loss(pred, target) + epsilon).sqrt()
