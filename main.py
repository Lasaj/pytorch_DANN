import torch
import train
import mnist
import mnistm
import covid_x
import old_covid_x
import models
import params
from datetime import datetime
from utils import visualize_more

# Training options
save_name = datetime.now().strftime("%y%m%d_%H%M")
discriminator_loss = params.discriminator_loss
encoder_type = params.encoder_type


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if params.data_type == 'mnist':
        source_train_loader, _, source_test_loader = mnist.get_source_dataloaders(encoder_type)
        target_train_loader, _, target_test_loader = mnistm.get_test_dataloaders(encoder_type)
        out_features = 10
    elif params.data_type == 'covidx':
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = covid_x.get_data()
        out_features = 2
    else:
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = old_covid_x.get_data()
        out_features = 3

    print(f'Training {encoder_type} on {device}')
    if encoder_type == 'inceptionv3':
        encoder = models.get_iv3().to(device)
        classifier = models.Classifier(in_features=2048, out_features=out_features).to(device)
        discriminator = models.Discriminator(in_features=2048).to(device)
    elif encoder_type == 'densenet':
        encoder = models.get_densenet(params.use_xrv_weights).to(device)
        classifier = models.Classifier(in_features=1024, out_features=out_features).to(device)
        discriminator = models.Discriminator(in_features=1024).to(device)
    elif encoder_type == 'resnet':
        encoder = models.get_resnet().to(device)
        classifier = models.Classifier(in_features=2048, out_features=out_features).to(device)
        discriminator = models.Discriminator(in_features=2048).to(device)
    else:
        encoder = models.Extractor().to(device)
        classifier = models.Classifier().to(device)
        discriminator = models.Discriminator().to(device)

    train.source_only(device, encoder, classifier, source_train_loader, source_test_loader, target_train_loader,
                      target_test_loader, save_name + '_' + encoder_type)
    if params.experiment_type == 'dann':
        train.dann(device, encoder, classifier, discriminator, discriminator_loss, source_train_loader,
                   source_test_loader, target_train_loader, target_test_loader, save_name + '_' + encoder_type)

    # encoder.load_state_dict(torch.load('./trained_models/old_covidx/encoder_source_230614_1755_densenet.pt', map_location=device))
    # classifier.load_state_dict(torch.load('./trained_models/old_covidx/classifier_source_230614_1755_densenet.pt', map_location=device))
    # visualize_more(device, encoder, 'source', save_name, classifier)

    # encoder.load_state_dict(torch.load('./trained_models/old_covidx/encoder_dann_230614_1755_densenet.pt', map_location=device))
    # classifier.load_state_dict(torch.load('./trained_models/old_covidx/classifier_dann_230614_1755_densenet.pt', map_location=device))
    # visualize_more(device, encoder, 'dann', save_name, classifier)


if __name__ == "__main__":
    main()
