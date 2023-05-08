import torch
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
# from dython.nominal import associations

root_dir = '../covid_x/'
train_data = f'{root_dir}train.txt'
test_data = f'{root_dir}test.txt'


class CovidXDataset(Dataset):
    def __init__(self, data_csv, root_dir, transform=None):
        self.data_csv = data_csv
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_csv.iloc[idx, 1])
        image = Image.open(img_name)
        label = self.data_csv.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)

        return image, label


def make_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight


def prepare_dfs():
    train_df = pd.read_csv(train_data, sep=" ", header=None)
    headers = ['id', 'file', 'label', 'source']
    train_df.columns = headers
    train_df.label = train_df.label.astype('category')
    train_df.replace(['negative', 'positive'], [0, 1], inplace=True)

    train_source = train_df[train_df['source'] != 'rsna'].reset_index(drop=True)
    train_target = train_df[train_df['source'] == 'rsna'].reset_index(drop=True)
    train_source.columns = headers
    train_target.columns = headers

    test_df = pd.read_csv(test_data, sep=" ", header=None)
    test_df.columns = headers
    test_df.label = test_df.label.astype('category')
    test_df.replace(['negative', 'positive'], [0, 1], inplace=True)

    test_source = test_df[test_df['source'] != 'rsna'].reset_index(drop=True)
    test_target = test_df[test_df['source'] == 'rsna'].reset_index(drop=True)
    test_source.columns = headers
    test_target.columns = headers

    return train_source, train_target, test_source, test_target


def prepare_dls(train_transform, val_transform, train_batch_size, test_batch_size, n_classes=2):
    train_source, train_target, test_source, test_target = prepare_dfs()

    # Get Datasets and DataLoaders for each split
    train_source_ds = CovidXDataset(train_source, root_dir + 'train/', transform=train_transform)
    train_target_ds = CovidXDataset(train_target, root_dir + 'train/', transform=train_transform)
    test_source_ds = CovidXDataset(test_source, root_dir + 'test/', transform=val_transform)
    test_target_ds = CovidXDataset(test_target, root_dir + 'test/', transform=val_transform)

    # train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    # val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    # test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
    # test_ds = copy.deepcopy(train_ds)

    # data = {}
    #
    # for data_set in [train_source_ds, train_target_ds, test_source_ds, test_target_ds]:
    #     labels = train_source_ds.data_csv.label
    #     weights = make_weights_for_balanced_classes(labels, n_classes)
    #     weights = torch.DoubleTensor(weights)
    #     sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    #     data[data_set] = DataLoader(data_set, batch_size=train_batch_size, sampler=sampler, drop_last=True)

    source_labels = train_source_ds.data_csv.label
    source_weights = make_weights_for_balanced_classes(source_labels, n_classes)
    source_weights = torch.DoubleTensor(source_weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(source_weights, len(source_weights))
    train_source_dl = DataLoader(train_source_ds, batch_size=train_batch_size, sampler=sampler, drop_last=True)

    # source_dl = DataLoader(source_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    # target_dl = DataLoader(target_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    # train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    # val_dl = DataLoader(val_ds, batch_size=train_batch_size, shuffle=False, drop_last=False)
    # test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)

    # train_source_dl = DataLoader(train_source_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    train_target_dl = DataLoader(train_target_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    test_source_dl = DataLoader(test_source_ds, batch_size=test_batch_size, shuffle=True, drop_last=False)
    test_target_dl = DataLoader(test_target_ds, batch_size=test_batch_size, shuffle=True, drop_last=False)

    # train_source_dl = data[train_source_ds]
    # train_target_dl = data[train_target_ds]
    # test_source_dl = data[test_source_ds]
    # test_target_dl = data[test_target_ds]

    return train_source_dl, train_target_dl, test_source_dl, test_target_dl


def get_data(transform=True):
    # train_df, val_df, test_df = prepare_dfs(data_csv)
    train_transform, val_transform = None, None

    if transform:
        train_transform = transforms.Compose([transforms.Resize((299, 299)),
                                              # transforms.CenterCrop((299, 299)),
                                              # transforms.RandomHorizontalFlip(),
                                              # transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(20),
                                              transforms.Grayscale(),
                                              # transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                              transforms.ToTensor(),
                                              # transforms.Normalize(img_mean, img_std)
                                              ])
        val_transform = transforms.Compose([transforms.Resize((299, 299)),
                                            transforms.Grayscale(),
                                            # transforms.CenterCrop((299, 299)),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # transforms.RandomRotation(20),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(img_mean, img_std)
                                            ])

    return prepare_dls(train_transform, val_transform, 32, 32)


def show_data_dist():
    headers = ['id', 'file', 'label', 'source']
    df = pd.read_csv(train_data, sep=" ", header=None)
    df2 = pd.read_csv(test_data, sep=" ", header=None)
    df = pd.concat([df, df2])
    df.columns = headers
    assoc_df = df.drop(['id', 'file'], axis=1)
    print(df.head())

    # assoc = associations(assoc_df, nom_nom_assoc='theil', figsize=(7, 7), cmap='RdBu',
    #                      title="Correlation in CovidX",
    #                      filename="../datasets/covid_x/covid_x_correlation.png")

    dd = pd.crosstab(df.label, df.source, normalize='columns')
    sns.heatmap(dd, annot=True, fmt='.3f', cmap='Blues')
    plt.title("Representation of dx in each sex")
    plt.show()
    plt.clf()

    sns.countplot(df, x='source', hue='label')
    plt.show()


def main():

    source_train_loader, target_train_loader, source_test_loader, target_test_loader = get_data()
    # print(len(train_source_dl), len(train_target_dl), len(test_source_dl), len(test_target_dl))
    # for s_img, s_label in test_target_dl:
    #     print(s_img.shape)
    #     print(s_label)
    # print(type(s_img))
    # print(type(s_label))
    print(len(source_train_loader), len(target_train_loader))
    # for i, s in enumerate(source_train_loader):
    #     print(i)
    for i, t in enumerate(target_train_loader):
        print(i)
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        source_image, source_label = source_data
        print(source_label)

    # show_data_dist()


if __name__ == '__main__':
    main()
