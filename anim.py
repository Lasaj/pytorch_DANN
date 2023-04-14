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


def get_data(device):
    # Draw 512 samples in test_data
    source_test_loader = mnist.mnist_test_loader
    target_test_loader = mnistm.mnistm_test_loader

    # Get source_test samples
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= 16:  # to get only 512 samples
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
        if i >= 16:
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

    # img_files = []
    # for i, img in enumerate(combined_img_list):
    #     file_name = f'./{training_mode}_imgs/{i}.png'
    #     save_image(img, file_name)
    #     img_files.append(file_name)

    source_domain_list = torch.zeros(512).type(torch.LongTensor)
    target_domain_list = torch.ones(512).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).to(device)

    return combined_label_list, combined_img_list, combined_domain_list


def perform_tsne(device, features, imgs, labels, domains, save_name, base_fit):
    encoder = model.Extractor().to(device)
    tsne = TSNE(perplexity=30, n_components=2, n_iter=3000)

    base = features[0] if base_fit == 'first' else features[-1]
    loaded = torch.load(base, map_location=device)
    encoder.load_state_dict(loaded)
    combined_features = encoder(imgs)
    embedding = tsne.fit(combined_features.detach().cpu().numpy())

    for epoch, epoch_features in enumerate(features):
        encoder.load_state_dict(torch.load(epoch_features, map_location=device))

        print(f"TSNE epoch {epoch}")
        combined_feature = encoder(imgs)  # combined_feature : 1024,2352
        dann_tsne = embedding.transform(combined_feature.detach().cpu().numpy())
        ep_str = '0' + str(epoch) if epoch < 10 else str(epoch)
        print('Draw plot')
        plot_embedding(dann_tsne, labels, domains, f'anim/{base_fit}/', save_name + ep_str)
        # create_bokeh(dann_tsne, combined_label_list, combined_domain_list, img_files, f"{save_name}_{training_mode}")


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
#         print('Draw plot')
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


def main():
    base_fit = 'last'  # 'last' or 'first'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_features = get_features('./trained_models/anim/')
    labels, imgs, domains = get_data(device)
    perform_tsne(device, all_features, imgs, labels, domains, 'anim', base_fit)
    make_gif(f'./saved_plot/anim/{base_fit}/')


if __name__ == "__main__":
    main()
