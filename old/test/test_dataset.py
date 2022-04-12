from lire.data_tools.dataset import MSMarco

# where are or will be downloaded the corpus
data_folder = '/media/gerald/00B1B02B44A76AB2/CPD/data'
# what split to use
split = 'train'
# laod the dataset with triplet output (query, positive, negative)
dataset = MSMarco.MSMarcoPassageRankingDataset(data_folder, download=True, split=split, storage='full', getter='triplet')