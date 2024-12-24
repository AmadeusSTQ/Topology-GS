import torch
import time
from torchvision.transforms import Grayscale
# from topologylayer.nn.alpha_dionysus import AlphaLayer
from topologylayer.nn.alpha import AlphaLayer
from topologylayer.nn.features import TopKBarcodeLengths


def rgb_to_grayscale(x):
    return Grayscale(num_output_channels=1)(x)


class PersistLoss(torch.nn.Module):
    def __init__(self, dims=[0, 1, 2], ks=[100, 20, 10]):
        super(PersistLoss, self).__init__()
        self.layer = AlphaLayer(maxdim=dims[-1])

        self.dims, self.ks = dims, ks
        assert isinstance(dims, list) and isinstance(ks, list) and len(dims) == len(ks), 'check input please'
        for i, dim in enumerate(dims):
            self.add_module('feature_{}'.format(str(i)), TopKBarcodeLengths(dim, ks[i]))

        self.feature_loss = torch.nn.MSELoss()

    def forward(self, x1, x2, diag_gt=None):
        loss = torch.tensor(0).to(torch.float).to(x1.device)

        # x1: cxhxw pred_image
        # x2: cxhxw gt_image

        c, h, w = x1.shape
        x1 = x1.permute(1, 2, 0).contiguous().view(-1, c)
        diag1 = self.layer(x1.cpu())

        if diag_gt is None:
            x2 = x2.permute(1, 2, 0).contiguous().view(-1, c)
            diag2 = self.layer(x2.cpu())
        else:
            diag2 = diag_gt

        for i, dim in enumerate(self.dims):
            feature1 = self.__getattr__('feature_{}'.format(str(i)))(diag1)
            feature2 = self.__getattr__('feature_{}'.format(str(i)))(diag2)

            feature1 = feature1.to(x2.device)
            feature2 = feature2.to(x2.device)

            # compute loss between topological features
            loss += self.feature_loss(feature1, feature2)

        return loss, diag2


if __name__ == "__main__":
    h, w = (60, 100)

    x1 = torch.randn(3, h, w, requires_grad=True).cuda()
    x2 = torch.randn(3, h, w, requires_grad=True).cuda()
    ps = PersistLoss().cuda()

    start_time = time.time()
    persist_loss = ps(x1, x2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print('persistence loss: ', persist_loss.item())
