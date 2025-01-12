import torch.nn.functional as F
import torch
import mnist
import mnistm
import covid_x
import itertools
import os

import old_covid_x
import params
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Function
# from sklearn.manifold import TSNE
from openTSNE import TSNE
from visualiser import create_bokeh
from torchvision.utils import save_image


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer


def my_kl(predicted, target):
    """
    Calculate KL divergence for domain loss function
    """
    target = F.one_hot(target.long(), num_classes=2)
    return -(target * torch.log(predicted.clamp_min(1e-7))).sum(dim=1).mean() - \
        -1 * (target.clamp(min=1e-7) * torch.log(target.clamp(min=1e-7))).sum(dim=1).mean()


def kl(pred, target):
    target = F.one_hot(target, num_classes=2)
    # print(pred[0], target)
    # exit()
    return torch.sum(target * torch.log(target.clamp(min=1e-7) / pred.clamp(min=1e-7)), dim=1).mean()


def one_hot_embedding(labels, num_classes=10):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def save_model(encoder, classifier, discriminator, training_mode, save_name):
    print('Save models ...')

    save_folder = 'trained_models'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.save(encoder.state_dict(), 'trained_models/encoder_' + str(training_mode) + '_' + str(save_name) + '.pt')
    torch.save(classifier.state_dict(),
               'trained_models/classifier_' + str(training_mode) + '_' + str(save_name) + '.pt')

    if training_mode == 'dann':
        torch.save(discriminator.state_dict(),
                   'trained_models/discriminator_' + str(training_mode) + '_' + str(save_name) + '.pt')

    print('Model is saved !!!')


def plot_embedding(X, y, d, training_mode, save_name, axis_limits=None):
    # x_min, x_max = np.min(X, 0), np.max(X, 0)
    # X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    if axis_limits is not None:
        plt.xlim(axis_limits['min_x'], axis_limits['max_x'])
        plt.ylim(axis_limits['min_y'], axis_limits['max_y'])

    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)
        else:
            colors = (1.0, 0.0, 0.0, 1.0)
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if save_name is not None:
        plt.title(save_name)

    save_folder = 'saved_plot'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    fig_name = 'saved_plot/' + str(training_mode) + '_' + str(save_name) + '.png'
    plt.savefig(fig_name)
    print('{} is saved'.format(fig_name))
    plt.close()


def visualize(device, encoder, training_mode, save_name):
    # Draw 512 samples in test_data
    # source_test_loader = mnist.mnist_test_loader
    # target_test_loader = mnistm.mnistm_test_loader
    visualise_batches = params.visualise_batches
    batch_size = params.batch_size

    if encoder.__class__.__name__ == 'Inception3':
        encoder_type = 'inceptionv3'
    else:
        encoder_type = 'extractor'

    if params.data_type == 'mnist':
        source_train_loader, _, source_test_loader = mnist.get_source_dataloaders(encoder_type)
        target_train_loader, _, target_test_loader = mnistm.get_test_dataloaders(encoder_type)
    elif params.data_type == 'covid':
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = covid_x.get_data()
    else:
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = old_covid_x.get_data()

    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= visualise_batches:
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        img = torch.cat((img, img, img), 1)
        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    if encoder.__class__.__name__ == 'Inception3':
        source_img_list = source_img_list.view(-1, 3, 299, 299)
    else:
        source_img_list = source_img_list.view(-1, 3, 28, 28)

    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= visualise_batches:
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        if params.data_type != 'mnist':
            img = torch.cat((img, img, img), 1)
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    if encoder.__class__.__name__ == 'Inception3':
        target_img_list = target_img_list.view(-1, 3, 299, 299)
    else:
        target_img_list = target_img_list.view(-1, 3, 28, 28)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)

    combined_img_list = torch.cat((source_img_list, target_img_list), 0)
    img_files = []
    for i, img in enumerate(combined_img_list):
        # plt.imshow(img.permute(1, 2, 0))
        # plt.show()
        file_name = f'./{training_mode}_imgs/{i}.png'
        save_image(img, file_name)
        img_files.append(file_name)

    source_domain_list = torch.zeros(visualise_batches * batch_size).type(torch.LongTensor)
    target_domain_list = torch.ones(visualise_batches * batch_size).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)

    print("Extract features to draw T-SNE plot...")
    combined_features = encoder(combined_img_list)  # combined_features
    if encoder.__class__.__name__ == 'Inception3':
        combined_features = combined_features[0]

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_features.detach().cpu().numpy())
    print(dann_tsne[0])

    print('Draw plot ...')
    save_name = save_name + '_' + str(training_mode)
    if params.data_type == 'mnist':
        plot_embedding(dann_tsne, combined_label_list, combined_domain_list, training_mode, save_name)
    create_bokeh(dann_tsne, combined_label_list, combined_domain_list, img_files, f"{save_name}_{training_mode}")


def visualize_input(device):
    visualise_batches = params.visualise_batches

    source_test_loader = mnist.mnist_test_loader
    target_test_loader = mnistm.mnistm_test_loader

    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= visualise_batches:
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    source_img_list = source_img_list.view(-1, 3, 28, 28)

    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= visualise_batches:
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    target_img_list = target_img_list.view(-1, 3, 28, 28)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_img_list = torch.cat((source_img_list, target_img_list), 0)

    source_domain_list = torch.zeros(512).type(torch.LongTensor)
    target_domain_list = torch.ones(512).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)

    print("Extract features to draw T-SNE plot...")
    combined_feature = combined_img_list  # combined_feature : 1024,3,28,28
    combined_feature = combined_feature.view(1024, -1)  # flatten
    # print(type(combined_feature), combined_feature.shape)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
    print('Draw plot ...')
    save_name = 'input_tsne_plot'
    plot_embedding(dann_tsne, combined_label_list, combined_domain_list, 'input', 'mnist_n_mnistM')


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def set_model_mode(mode='train', models=None):
    for model in models:
        if mode == 'train':
            model.train()
        else:
            model.eval()


def visualize_more(device, encoder, training_mode, save_name, classifier=None):
    # Draw 512 samples in test_data
    visualise_batches = params.visualise_batches
    batch_size = params.batch_size

    if encoder.__class__.__name__ == 'Inception3':
        encoder_type = 'inceptionv3'
    else:
        encoder_type = 'extractor'

    if params.data_type == 'mnist':
        source_train_loader, _, source_test_loader = mnist.get_source_dataloaders(encoder_type)
        target_train_loader, _, target_test_loader = mnistm.get_test_dataloaders(encoder_type)
    elif params.data_type == 'covid_x':
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = covid_x.get_data()
    else:
        source_train_loader, target_train_loader, source_test_loader, target_test_loader = old_covid_x.get_data()

    tsne = TSNE(perplexity=10.33, n_components=2, n_iter=3000)
    combined_label_list, combined_domain_list, combined_id_list, img_files, embeddings, preds = [], [], [], [], [], []

    if params.visualise_data_set == "test":
        source_loader = source_test_loader
        target_loader = target_test_loader
    else:
        source_loader = source_train_loader
        target_loader = target_train_loader

    source_csv = source_loader.dataset.data_csv
    target_csv = target_loader.dataset.data_csv
    combined_csv = pd.concat([source_csv, target_csv])

    for batch, (source_data, target_data) in enumerate(zip(source_loader, target_loader)):
        if batch >= visualise_batches:
            break
        if len(source_data) <= 0 or len(target_data) <= 0:
            break

        # Get source_test samples
        source_label_list = []
        source_img_list = []
        source_id_list = []
        if params.data_type == 'mnist':
            source_img, source_label = source_data
        else:
            source_img, source_label, source_id = source_data
            source_id_list.append(source_id)

        source_label = source_label.numpy()
        source_img = source_img.to(device)
        source_img = torch.cat((source_img, source_img, source_img), 1)
        source_label_list.append(source_label)
        source_img_list.append(source_img)

        source_img_list = torch.stack(source_img_list)
        if encoder.__class__.__name__ == 'Inception3':
            source_img_list = source_img_list.view(-1, 3, 299, 299)
        else:
            source_img_list = source_img_list.view(-1, 3, 224, 224)
            # source_img_list = source_img_list.view(-1, 3, 28, 28)

        # Get target_test samples
        target_label_list = []
        target_img_list = []
        target_id_list = []

        if params.data_type == 'mnist':
            target_img, target_label = target_data
        else:
            target_img, target_label, target_id = target_data
            target_id_list.append(target_id)

        target_label = target_label.numpy()
        target_img = target_img.to(device)
        if params.data_type != 'mnist':
            target_img = torch.cat((target_img, target_img, target_img), 1)
        target_label_list.append(target_label)
        target_img_list.append(target_img)

        target_img_list = torch.stack(target_img_list)
        if encoder.__class__.__name__ == 'Inception3':
            target_img_list = target_img_list.view(-1, 3, 299, 299)
        else:
            target_img_list = target_img_list.view(-1, 3, 224, 224)
            # target_img_list = target_img_list.view(-1, 3, 28, 28)

        # Stack source_list + target_list
        batch_label_list = list(source_label_list[0])
        batch_label_list.extend(list(target_label_list[0]))

        batch_img_list = torch.cat((source_img_list, target_img_list), 0)
        # for i, img in enumerate(batch_img_list):
        #     file_name = f'./{training_mode}_imgs/{batch}_{i}.png'
        #     save_image(img, file_name)
        #     img_files.append(file_name)

        # source_domain_list = torch.zeros(visualise_batches * batch_size).type(torch.LongTensor)
        # target_domain_list = torch.ones(visualise_batches * batch_size).type(torch.LongTensor)
        # batch_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)
        source_domain_list = [0] * len(source_img_list)
        target_domain_list = [1] * len(target_img_list)
        batch_domain_list = source_domain_list + target_domain_list

        if params.data_type != 'mnist':
            batch_id_list = list(source_id_list[0]) + list(target_id_list[0])
            combined_id_list.extend(batch_id_list)

        print("Extract features to draw T-SNE plot...")
        batch_features = encoder(batch_img_list)  # combined_features
        if encoder.__class__.__name__ == 'Inception3':
            batch_features = batch_features[0]

        if classifier is not None:
            batch_output = classifier(batch_features)
            batch_pred = batch_output.data.max(1, keepdim=True)[1]
            preds.extend(batch_pred)

        combined_label_list.extend(batch_label_list)
        combined_domain_list.extend(batch_domain_list)

        if batch == 0:
            embedding = tsne.fit(batch_features.detach().cpu().numpy())
        dann_tsne = embedding.transform(batch_features.detach().cpu().numpy())
        embeddings.extend(dann_tsne)

        del(batch_features)

    combined_domain_list = [int(x) for x in combined_domain_list]
    preds = [int(x) for x in preds]
    embeddings = np.array([[x[0], x[1]] for x in embeddings])

    print('Draw plot ...')
    save_name = save_name + '_' + str(training_mode)
    if params.data_type == 'mnist':
        plot_embedding(embeddings, combined_label_list, combined_domain_list, training_mode, save_name)
    else:
        create_bokeh(embeddings, combined_label_list, combined_domain_list, combined_id_list, combined_csv, preds, f"{save_name}_{training_mode}")
