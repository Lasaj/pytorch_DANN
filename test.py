import torch
import wandb
import numpy as np
from utils import set_model_mode


def tester(device, encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode):
    print("Model test ...")

    encoder.to(device)
    classifier.to(device)
    set_model_mode('eval', [encoder, classifier])

    if training_mode == 'dann':
        discriminator.to(device)
        set_model_mode('eval', [discriminator])
        domain_correct = 0

    source_correct = 0
    target_correct = 0

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        source_image, source_label = source_data
        # source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
        source_image, source_label = source_image.to(device), source_label.to(device)
        source_feature = encoder(source_image)
        # if encoder.__class__.__name__ == 'Inception3':
        #     source_feature = source_feature[0]
        source_output = classifier(source_feature)
        source_pred = source_output.data.max(1, keepdim=True)[1]
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).cpu().sum()

        # 2. Target input -> Target Classification
        target_image, target_label = target_data
        # target_image = torch.cat((target_image, target_image, target_image), 1)  # MNIST convert to 3 channel
        target_image, target_label = target_image.to(device), target_label.to(device)
        target_feature = encoder(target_image)
        # if encoder.__class__.__name__ == 'Inception3':
        #     target_feature = target_feature[0]
        target_output = classifier(target_feature)
        target_pred = target_output.data.max(1, keepdim=True)[1]
        target_correct += target_pred.eq(target_label.data.view_as(target_pred)).cpu().sum()

        if training_mode == 'dann':
            # 3. Combined input -> Domain Classificaion
            combined_image = torch.cat((source_image, target_image), 0)  # 64 = (S:32 + T:32)
            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).to(device)
            domain_feature = encoder(combined_image)
            # if encoder.__class__.__name__ == 'Inception3':
            #     domain_feature = domain_feature[0]
            domain_output = discriminator(domain_feature, alpha)
            domain_pred = domain_output.data.max(1, keepdim=True)[1]
            domain_correct += domain_pred.eq(domain_combined_label.data.view_as(domain_pred)).cpu().sum()

    if training_mode == 'dann':
        source_accuracy = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_accuracy = 100. * target_correct.item() / len(target_test_loader.dataset)
        domain_accuracy = 100. * domain_correct.item() / (
                len(source_test_loader.dataset) + len(target_test_loader.dataset))
        print("Test Results on DANN :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'
              'Domain Accuracy: {}/{} ({:.2f}%)\n'.format(
                source_correct, len(source_test_loader.dataset), source_accuracy,
                target_correct, len(target_test_loader.dataset), target_accuracy,
                domain_correct, len(source_test_loader.dataset) + len(target_test_loader.dataset), domain_accuracy))
        wandb.log({'dann_source_accuracy': source_accuracy, 'dann_target_accuracy': target_accuracy, 'dann_domain_accuracy': domain_accuracy})
    else:
        source_accuracy = 100. * source_correct.item() / len(source_test_loader.dataset)
        target_accuracy = 100. * target_correct.item() / len(target_test_loader.dataset)
        print("Test results on source_only :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'.format(
                source_correct, len(source_test_loader.dataset), source_accuracy,
                target_correct, len(target_test_loader.dataset), target_accuracy))
        wandb.log({'source_accuracy': source_accuracy, 'target_accuracy': target_accuracy})
