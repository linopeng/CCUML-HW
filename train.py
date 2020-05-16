from hw2.model.cnn import cnn_model
from hw2.config import cfg
from hw2.datasets.dataloader import MangoDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

import torch, os
import numpy as np

def train():
    model = cnn_model()

    valid_size = cfg.DATA.VALIDATION_SIZE
    epochs = cfg.MODEL.EPOCH
    lr = cfg.MODEL.LR
    weight_path = cfg.MODEL.OUTPUT_PATH
    use_cuda = cfg.DEVICE.CUDA
    gpu_id = cfg.DEVICE.GPU
    root_path = cfg.PATH.TRAIN_SET
    num_workers = cfg.DATA.NUM_WORKERS
    batch_size = cfg.DATA.TRAIN_BATCH_SIZE

    transform = transforms.Compose([transforms.Resize(cfg.DATA.RESIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(cfg.DATA.PIXEL_MEAN,
                                                         cfg.DATA.PIXEL_STD)])

    print(Path(root_path).joinpath('C1-P1_Train'))
    train_set = MangoDataset(Path(root_path).joinpath('C1-P1_Train'), transform)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers)

    valid_set = MangoDataset(Path(root_path).joinpath('C1-P1_Dev'), transform)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, num_workers=num_workers)

    if use_cuda:
        torch.cuda.set_device(gpu_id)
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.
        valid_loss = 0.
        correct = 0.

        for data, target in train_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        model.eval()
        for data, target in valid_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            correct += (output.max(1)[1] == target).sum()
            valid_loss += loss.item() * data.size(0)

        accuracy = 100. * correct / len(valid_loader.dataset)
        train_loss /= int(len(train_loader.dataset))
        valid_loss /= int(len(valid_loader.dataset))
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f} Valid Accuracy: {:.2f}%'.format(epoch, train_loss,
                                                                                                         valid_loss,
                                                                                                         accuracy))

    output_dir = "/".join(weight_path.split("/")[:-1])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), weight_path)
