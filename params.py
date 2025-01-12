# Change experiment parameters here #
batch_size = 32
epochs = 100
num_workers = 2
discriminator_loss = 'crossentropy'  # Available: 'crossentropy', 'kl'
encoder_type = 'densenet'  # Available: 'inceptionv3', 'extractor', 'densenet', 'resnet'
use_xrv_weights = True  # Weights trained by https://github.com/mlmed/torchxrayvision
experiment_type = 'dann'  # Available: 'dann', 'source_only'
data_type = 'covidx'  # Available: 'mnist', 'covidx', 'old_covidx'

# Change data parameters here #
covidx_dir = './covid_x/'
old_covidx_dir = './old_covid_x/'

# Change visualisation parameters here #
visualise_batches = 60  # n * batch_size for TSNE sample (might need to change batch_size to 16)
visualise_data_set = "test"  # Available: 'train', 'test'
