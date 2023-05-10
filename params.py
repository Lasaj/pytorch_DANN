batch_size = 32
epochs = 100
num_workers = 2
discriminator_loss = 'crossentropy'  # Available: 'crossentropy', 'kl'
encoder_type = 'inceptionv3'  # Available: 'inceptionv3', 'extractor'
experiment_type = 'dann'  # Available: 'dann', 'source_only'
data_type = 'covidx'  # Available: 'mnist', 'covidx'

visualise_batches = 2  # n * batch_size for TSNE sample (16 works for MNIST)
