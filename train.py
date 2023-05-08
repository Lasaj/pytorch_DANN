import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
# import mnist
# import mnistm
from utils import save_model
from utils import visualize
from utils import set_model_mode
import params

# Source : 0, Target :1
# source_test_loader = mnist.mnist_test_loader
# target_test_loader = mnistm.mnistm_test_loader


def source_only(device, encoder, classifier, source_train_loader, source_test_loader, target_train_loader,
                target_test_loader, save_name):
    print("Source-only training")
    classifier_criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)

    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        # print(len(target_train_loader), len(source_train_loader))
        # exit()
        # for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        for batch_idx, source_data in enumerate(source_train_loader):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.to(device), source_label.to(device)  # 32

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            if encoder.__class__.__name__ == 'Inception3':
                source_feature = source_feature[0]

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image),
                                                                     len(source_train_loader.dataset),
                                                                     100. * batch_idx / len(source_train_loader),
                                                                     class_loss.item()))

        if (epoch + 1) % 1 == 0:
            ep_str = '0' + str(epoch) if epoch < 10 else str(epoch)
            save_model(encoder, classifier, None, 'source', save_name + '_' + ep_str)
            test.tester(device, encoder, classifier, None, source_test_loader, target_test_loader,
                        training_mode='source_only')

    save_model(encoder, classifier, None, 'source', save_name)
    visualize(device, encoder, 'source', save_name)


def dann(device, encoder, classifier, discriminator, loss_type, source_train_loader, source_test_loader,
         target_train_loader, target_test_loader, save_name):
    print("DANN training")

    classifier_criterion = nn.CrossEntropyLoss().to(device)
    discriminator_criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()) +
        list(discriminator.parameters()),
        lr=0.01,
        momentum=0.9)

    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = params.epochs * len(target_train_loader)

        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.to(device), source_label.to(device)
            target_image, target_label = target_image.to(device), target_label.to(device)
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            if encoder.__class__.__name__ == 'Inception3':
                combined_feature = combined_feature[0]
                source_feature = source_feature[0]


            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).to(device)
            if loss_type == 'kl':
                domain_loss = utils.kl(domain_pred, domain_combined_label)
            else:
                domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            domain_loss_weight = max(0, (epoch - 20) / params.epochs)

            total_loss = class_loss + (domain_loss * domain_loss_weight)
            total_loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % 50 == 0:
            if (batch_idx + 1) % 100 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(target_image), len(target_train_loader.dataset),
                    100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(),
                    domain_loss.item()))
                # visualize(device, encoder, 'dann', save_name + '_' + ep_str)

        if (epoch + 1) % 1 == 0:
            ep_str = '0' + str(epoch) if epoch < 10 else str(epoch)
            save_model(encoder, classifier, discriminator, 'dann', save_name + '_' + ep_str)
            test.tester(device, encoder, classifier, discriminator, source_test_loader, target_test_loader,
                        training_mode='dann')

    save_model(encoder, classifier, discriminator, 'dann', save_name)
    visualize(device, encoder, 'dann', save_name)
