import torch
import train
import mnist
import mnistm
import model
from datetime import datetime
from utils import get_free_gpu

start_time = datetime.now()
save_name = f'{start_time}'


def main():

    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    # for i in source_train_loader.dataset:
    #     if i[0].shape[0] != 3:
    #         print(i[0].shape)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    encoder = model.Extractor().to(device)
    classifier = model.Classifier().to(device)
    discriminator = model.Discriminator().to(device)

    train.source_only(device, encoder, classifier, source_train_loader, target_train_loader, save_name)
    train.dann(device, encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name)


if __name__ == "__main__":
    main()
