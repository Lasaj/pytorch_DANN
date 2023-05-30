batch_size = 16
epochs = 100
num_workers = 2
discriminator_loss = 'crossentropy'  # Available: 'crossentropy', 'kl'
encoder_type = 'inceptionv3'  # Available: 'inceptionv3', 'extractor'
experiment_type = 'dann'  # Available: 'dann', 'source_only'
data_type = 'covidx'  # Available: 'mnist', 'covidx', 'old_covidx'
covidx_dir = './covid_x/'
old_covidx_dir = './old_covid_x/'

visualise_batches = 30  # n * batch_size for TSNE sample (16 works for MNIST)
visualise_data_set = "train"  # Available: 'train', 'test'
