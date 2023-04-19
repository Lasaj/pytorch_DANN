import torch
import train
import mnist
import mnistm
import model
from datetime import datetime

# Training options
save_name = datetime.now().strftime("%y%m%d_%H%M") + '_IV3'
discriminator_loss = 'crossentropy'  # Available: 'crossentropy', 'kl'
encoder_type = 'inceptionv3'  # Available: 'inceptionv3', 'extractor'


def main():
    source_train_loader, _, source_test_loader = mnist.get_source_dataloaders(encoder_type)
    # target_train_loader = mnistm.mnistm_train_loader
    target_train_loader, _, target_test_loader = mnistm.get_test_dataloaders(encoder_type)

    # for i in source_train_loader.dataset:
    #     if i[0].shape[0] != 3:
    #         print(i[0].shape)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    if encoder_type == 'inceptionv3':
        encoder = model.get_iv3().to(device)
        classifier = model.Classifier(in_features=2048).to(device)
        discriminator = model.Discriminator(in_features=2048).to(device)
    else:
        encoder = model.Extractor().to(device)
        classifier = model.Classifier().to(device)
        discriminator = model.Discriminator().to(device)

    # train.source_only(device, encoder, classifier, source_train_loader, source_test_loader, target_train_loader,
    #                   target_test_loader, save_name)
    train.dann(device, encoder, classifier, discriminator, discriminator_loss, source_train_loader, source_test_loader,
               target_train_loader, target_test_loader, save_name)

    # encoder.load_state_dict(torch.load('./trained_models/encoder_source_230327_1204.pt', map_location=device))
    # visualize(device, encoder, 'source', save_name)


if __name__ == "__main__":
    main()
