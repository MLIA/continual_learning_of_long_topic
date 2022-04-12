'''The T5 ranking baseline.
The baseline implemented here rely on the paper
"Document Ranking with a Pretrained Sequence-to-Sequence Model"
published at EMNLP (2020).
'''
import logging
from lire.data_tools.dataset import MSMarco


from experiments import t5ranker
# Loading dataset
data_folder = '/local/gerald/CPD/data'
split = 'train'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

logging.info('Loading MSMarcoPassageRanking')
# loading the corpus using positive
dataset = MSMarco.MSMarcoPassageRankingDataset(data_folder,
                                               download=False,
                                               split=split,
                                               getter='triplet')

logging.info('Configure the experiment')

nb_queries = dataset.get_nb_annotated_queries()
options = t5ranker.default_options
options.batch_size = 20
options.device = 1
batch_size = options.batch_size

def f_iter():
    dataset.get_nb_queries() * 5

nb_iteration = f_iter
options.number_iteration_by_task = nb_iteration
options.save = '/local/gerald/T5RankerFineTunedMSMarco.pth'

# we train from 10 epochs according to number of training queries
my_experiment = t5ranker.T5TaskRanker(options, dataset)
my_experiment.run()
