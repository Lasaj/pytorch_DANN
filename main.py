import torch
import train
import mnist
import mnistm
import covid_x
import old_covid_x
import model
import params
from datetime import datetime
from utils import visualize_more

# Training options
save_name = datetime.now().strftime("%y%m%d_%H%M") + '_IV3'
discriminator_loss = params.discriminator_loss
encoder_type = params.encoder_type


def main():
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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')
    if encoder_type == 'inceptionv3':
        encoder = model.get_iv3().to(device)
        classifier = model.Classifier(in_features=2048, out_features=out_features).to(device)
        discriminator = model.Discriminator(in_features=2048).to(device)
    else:
        encoder = model.Extractor().to(device)
        classifier = model.Classifier().to(device)
        discriminator = model.Discriminator().to(device)

    train.source_only(device, encoder, classifier, source_train_loader, source_test_loader, target_train_loader,
                      target_test_loader, save_name)
    if params.experiment_type == 'dann':
        train.dann(device, encoder, classifier, discriminator, discriminator_loss, source_train_loader,
                   source_test_loader, target_train_loader, target_test_loader, save_name)

    # encoder.load_state_dict(torch.load('./trained_models/covidx_both/encoder_source_230512_0801_IV3.pt', map_location=device))
    # classifier.load_state_dict(torch.load('./trained_models/covidx_both/classifier_source_230512_0801_IV3.pt', map_location=device))
    # visualize_more(device, encoder, 'source', save_name, classifier)

    # encoder.load_state_dict(torch.load('./trained_models/covidx_both/encoder_dann_230512_0801_IV3.pt', map_location=device))
    # classifier.load_state_dict(torch.load('./trained_models/covidx_both/classifier_dann_230512_0801_IV3.pt', map_location=device))
    # visualize_more(device, encoder, 'dann', save_name, classifier)


if __name__ == "__main__":
    main()
