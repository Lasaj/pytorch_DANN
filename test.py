import torch
import numpy as np

import params
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
    source_preds = [0] * 3
    correct_preds = [0] * 3
    count_all = [0] * 3

    for batch_idx, (source_data, target_data) in enumerate(zip(source_test_loader, target_test_loader)):
        p = float(batch_idx) / len(source_test_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # 1. Source input -> Source Classification
        if params.data_type == 'mnist':
            source_image, source_label = source_data
        else:
            source_image, source_label, _ = source_data
        source_image = torch.cat((source_image, source_image, source_image), 1)
        source_image, source_label = source_image.to(device), source_label.to(device)
        source_feature = encoder(source_image)
        # if encoder.__class__.__name__ == 'Inception3':
        #     source_feature = source_feature[0]
        source_output = classifier(source_feature)
        # count number of each prediction
        a = torch.argmax(source_output, dim=1)
        a_pred = a.cpu().numpy()
        b_label = source_label.cpu().numpy()
        print(a_pred, b_label)
        for i, v in enumerate(a_pred):
            if b_label[i] == v:
                correct_preds[v] += 1

        unique, counts = torch.unique(source_label, return_counts=True)
        for i, v in enumerate(unique.cpu().numpy()):
            count_all[v] += counts.cpu().numpy()[i]

        print(unique, counts)
        print(correct_preds, count_all)
        a = [0] * 3
        for i, v in enumerate(correct_preds):
            if count_all[i] != 0:
                a[i] = v / count_all[i]

        print(f"Accuracy: Normal: {a[0]}, Pneumonia: {a[1]}, COVID: {a[2]}")

        for i, v in enumerate(unique):
            source_preds[v] += counts[i].cpu().numpy()
        source_pred = source_output.data.max(1, keepdim=True)[1]
        source_correct += source_pred.eq(source_label.data.view_as(source_pred)).cpu().sum()

        # 2. Target input -> Target Classification
        if params.data_type == 'mnist':
            target_image, target_label = target_data
        if params.data_type != 'mnist':
            target_image, target_label, _ = target_data
            target_image = torch.cat((target_image, target_image, target_image), 1)
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

    print(f"Predictions: {source_preds}")
    if training_mode == 'dann':
        print("Test Results on DANN :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'
              'Domain Accuracy: {}/{} ({:.2f}%)\n'.
        format(
            source_correct, len(source_test_loader.dataset),
            100. * source_correct.item() / len(source_test_loader.dataset),
            target_correct, len(target_test_loader.dataset),
            100. * target_correct.item() / len(target_test_loader.dataset),
            domain_correct, len(source_test_loader.dataset) + len(target_test_loader.dataset),
            100. * domain_correct.item() / (len(source_test_loader.dataset) + len(target_test_loader.dataset))
        ))
    else:
        print("Test results on source_only :")
        print('\nSource Accuracy: {}/{} ({:.2f}%)\n'
              'Target Accuracy: {}/{} ({:.2f}%)\n'.format(
            source_correct, len(source_test_loader.dataset),
            100. * source_correct.item() / len(source_test_loader.dataset),
            target_correct, len(target_test_loader.dataset),
            100. * target_correct.item() / len(target_test_loader.dataset)))
