import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import dataset.CD_dataset as dates
import cfgs.config as cfg
import torch.utils.data as Data


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


if __name__ == '__main__' :
    train_transform_det = dates.Compose([dates.Scale(cfg.TRANSFROM_SCALES), ])

    train_data = dates.Dataset(cfg.TRAIN_DATA_PATH, cfg.TRAIN_LABEL_PATH,
                               cfg.TRAIN_TXT_PATH, 'train', transform=True,
                               transform_med=train_transform_det)
    train_loader = Data.DataLoader(train_data, batch_size=cfg.BATCH_SIZE,
                                   shuffle=True, num_workers=4, pin_memory=True)

    mean, std = get_mean_and_std(train_loader)

    print(mean, std)