import imageio.v2 as imageio
# from sklearn.manifold import TSNE
from utils import plot_embedding
import torch
import mnist
import model
import mnistm
from glob import glob
import os
# import umap
# from umap import UMAP
from openTSNE import TSNE


"""
Script to run TSNE over saved features to produce an animation
"""


def get_data(device, encoder_type):
    # Draw 512 samples in test_data
    # source_test_loader = mnist.mnist_test_loader
    # target_test_loader = mnistm.mnistm_test_loader
    _, _, source_test_loader = mnist.get_source_dataloaders(encoder_type)
    _, _, target_test_loader = mnistm.get_test_dataloaders(encoder_type)

    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= 1:  # to get only 512 samples
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        img = torch.cat((img, img, img), 1)  # MNIST channel 1 -> 3
        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    if encoder_type == 'inceptionv3':
        source_img_list = source_img_list.view(-1, 3, 299, 299)
    else:
        source_img_list = source_img_list.view(-1, 3, 28, 28)


    # Get target_test samples
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= 8:
            break
        img, label = test_data
        label = label.numpy()
        img = img.to(device)
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    if encoder_type == 'inceptionv3':
        target_img_list = target_img_list.view(-1, 3, 299, 299)
    else:
        target_img_list = target_img_list.view(-1, 3, 28, 28)

    # Stack source_list + target_list
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_img_list = torch.cat((source_img_list, target_img_list), 0)

    # img_files = []
    # for i, img in enumerate(combined_img_list):
    #     file_name = f'./{training_mode}_imgs/{i}.png'
    #     save_image(img, file_name)
    #     img_files.append(file_name)

    source_domain_list = torch.zeros(512).type(torch.LongTensor)
    target_domain_list = torch.ones(512).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)

    return combined_label_list, combined_img_list, combined_domain_list


def perform_tsne(device, encoder, features, imgs, base_fit):
    # encoder = model.Extractor().to(device)
    encoder = encoder.to(device)
    tsne = TSNE(perplexity=30, n_components=2, n_iter=3000)

    base = features[0] if base_fit == 'first' else features[-1]
    loaded = torch.load(base, map_location=device)
    encoder.load_state_dict(loaded)
    combined_features = encoder(imgs)
    if encoder.__class__.__name__ == 'Inception3':
        combined_features = combined_features[0]
    embedding = tsne.fit(combined_features.detach().cpu().numpy())

    axis_limits = {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

    all_embeddings = []
    for epoch, epoch_features in enumerate(features):
        if epoch > 99:
            break
        encoder.load_state_dict(torch.load(epoch_features, map_location=device))

        print(f"TSNE epoch {epoch}")
        combined_feature = encoder(imgs)  # combined_feature : 1024,2352
        dann_tsne = embedding.transform(combined_feature.detach().cpu().numpy())

        all_embeddings.append(dann_tsne)
        # x_min = min(dann_tsne[:, 0])
        # x_max = (max(dann_tsne[:, 0]) - x_min) / (max(dann_tsne[:, 0]) - x_min)
        # X = (X - x_min) / (x_max - x_min)
        axis_limits['min_x'] = min(axis_limits['min_x'], min(dann_tsne[:, 0]))
        axis_limits['max_x'] = max(axis_limits['max_x'], max(dann_tsne[:, 0]))
        axis_limits['min_y'] = min(axis_limits['min_y'], min(dann_tsne[:, 1]))
        axis_limits['max_y'] = max(axis_limits['max_y'], max(dann_tsne[:, 1]))

        # plot_embedding(dann_tsne, labels, domains, f'anim/{base_fit}/', save_name + ep_str)
        # create_bokeh(dann_tsne, combined_label_list, combined_domain_list, img_files, f"{save_name}_{training_mode}")

    return all_embeddings, axis_limits


def make_plots(all_embeddings, axis_limits, labels, domains, save_name, base_fit):
    for epoch, embedding in enumerate(all_embeddings):
        ep_str = '0' + str(epoch) if epoch < 10 else str(epoch)
        plot_embedding(embedding, labels, domains, f'{save_name}/{base_fit}/', save_name + ep_str, axis_limits=axis_limits)


# def perform_umap(device, features, imgs, labels, domains, save_name):
#     encoder = model.Extractor().to(device)
#     trans = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42)
#
#     last_features = features[-1]
#     loaded = torch.load(last_features, map_location=device)
#     encoder.load_state_dict(loaded)
#     combined_features = encoder(imgs)
#     trans = trans.fit(combined_features.detach().cpu().numpy())
#
#     for epoch, epoch_features in enumerate(features):
#         encoder.load_state_dict(torch.load(epoch_features, map_location=device))
#
#         print(f"TSNE epoch {epoch}")
#         combined_feature = encoder(imgs)  # combined_feature : 1024,2352
#         embedding = trans.transform(combined_feature.detach().cpu().numpy())
#
#         plot_embedding(embedding, labels, domains, 'anim/', save_name + str(epoch))
#         # create_bokeh(embedding, combined_label_list, combined_domain_list, img_files, f"{save_name}_{training_mode}")
#

def get_features(location):
    features = glob(os.path.join(location, "*"))
    features = sorted(features)
    return features


def make_gif(location):
    images = []
    for filename in sorted(os.listdir(location)):
        images.append(imageio.imread(os.path.join(location, filename)))
    imageio.mimsave(os.path.join(location, 'anim.gif'), images, duration=0.5, loop=1)
    print(f"Saved gif to {location}")


def main():
    encoder_type = "inceptionv3"
    location = "iv3_anim"
    # base_fit = 'last'  # 'last' or 'first'
    # base_fit = 'first'  # 'last' or 'first'
    if encoder_type == "inceptionv3":
        encoder = model.get_iv3()
    else:
        encoder = model.Extractor()

    for base_fit in ['first', 'last']:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        all_features = get_features(f'./trained_models/{location}/')
        print(f"Got {len(all_features)} features")
        labels, imgs, domains = get_data(device, encoder_type)
        all_embeddings, axis_limits = perform_tsne(device, encoder, all_features, imgs, base_fit)
        make_plots(all_embeddings, axis_limits, labels, domains, location, base_fit)
        # if folder doesn't exist, then create it
        if not os.path.exists(f'./saved_plot/{location}/{base_fit}/'):
            os.makedirs(f'./saved_plot/{location}/{base_fit}/')
        make_gif(f'./saved_plot/{location}/{base_fit}/')


if __name__ == "__main__":
    main()
