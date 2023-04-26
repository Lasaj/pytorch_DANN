import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from dython.nominal import associations


data = '../datasets/covid_x/train.txt'

def prepare_dls(train_dir, val_dir, test_dir, train_transform, val_transform, train_batch_size, test_batch_size):
    # Get Datasets and DataLoaders for each split
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=val_transform)
    # test_ds = copy.deepcopy(train_ds)

    # Get weights for training set
    # weights = make_weights_for_balanced_classes(train_ds, n_classes)
    # weights = torch.DoubleTensor(weights)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # train_dl = DataLoader(train_ds, batch_size=train_batch_size, sampler=sampler)
    # val_dl = DataLoader(val_ds, batch_size=train_batch_size, sampler=sampler)

    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=train_batch_size, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)

    return train_dl, val_dl, test_dl


def get_data(transform=True):
    # train_df, val_df, test_df = prepare_dfs(data_csv)
    train_transform, val_transform = None, None

    if transform:
        train_transform = transforms.Compose([transforms.Resize((299, 299)),
                                              # transforms.CenterCrop((299, 299)),
                                              # transforms.RandomHorizontalFlip(),
                                              # transforms.RandomVerticalFlip(),
                                              transforms.RandomRotation(20),
                                              # transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                              transforms.ToTensor(),
                                              # transforms.Normalize(img_mean, img_std)
                                              ])
        val_transform = transforms.Compose([transforms.Resize((299, 299)),
                                            # transforms.CenterCrop((299, 299)),
                                            # transforms.RandomHorizontalFlip(),
                                            # transforms.RandomVerticalFlip(),
                                            # transforms.RandomRotation(20),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(img_mean, img_std)
                                            ])

    return prepare_dls(train_transform, val_transform)

headers = ['id', 'file', 'label', 'source']
df = pd.read_csv(data, sep=" ", header=None)
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

# sns.histplot(x=df['source'], hue=df['label'], multiple="stack", kde=True,
#              stat='density', shrink=0.8, common_norm=False, fill=True)
# plt.title('Normalised Diagnosis by Age')
# plt.savefig("./testing/results/data_exp/dx_age_bars.png")
# plt.show()
# plt.clf()

sns.countplot(df, x='source', hue='label')
plt.show()

