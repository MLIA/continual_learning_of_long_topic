'''The T5 ranking baseline.
The baseline implemented here rely on the paper
"Document Ranking with a Pretrained Sequence-to-Sequence Model"
published at EMNLP (2020).
'''
import torch
from torch.cuda.amp import autocast
import logging
from lire.data_tools.dataset import MSMarco
import tqdm
import torch.utils.data as data_utils
from experiments import t5ranker
# Loading dataset
data_folder = '/local/gerald/CPD/data'
split = 'dev'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

logging.info('Loading MSMarcoPassageRanking')
# loading the corpus using positive
dataset = MSMarco.MSMarcoPassageRankingDataset(data_folder,
                                               download=False,
                                               split=split,
                                               getter='top1000', 
                                               subset_size=5)
dataset.set_current_task_by_id = lambda x : 1 

def fnull():
    return 1
dataset.get_nb_tasks = fnull

logging.info('Configure the experiment')

nb_queries = dataset.get_nb_annotated_queries()
options = t5ranker.default_options
options.batch_size = 32
options.device = 0
batch_size = options.batch_size
nb_iteration = nb_queries * 9
options.number_iteration_by_task = int(nb_iteration)
options.save = '/net/sledge/gerald/T5RankerBaselineMSMarco.pth'
print(options)
logging.info("Number of iteration considering the batch size: " +
             str(nb_iteration))
# we train from 10 epochs according to number of training queries
my_experiment = t5ranker.T5TaskRanker(options, dataset)
logging.info("Loading check point")
state_dict = torch.load('/net/sledge/gerald/T5RankerBaselineMSMarco.pth', map_location='cpu')
my_experiment.state_dict = state_dict
dataset.set_output_transformation(lambda x : x[0] + x[1])
my_experiment.options.batch_size = 32
prediction_set = {}

dataloader =\
    data_utils.DataLoader(dataset,
                            batch_size=my_experiment.options.batch_size,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

with torch.no_grad():
    with autocast(enabled=True):
        for it, (q_id, d_id, input_tokens) in zip(tqdm.trange(len(dataloader)),(dataloader)):
            positive_index =\
                my_experiment.tokenizer(input_tokens, return_tensors="pt", padding=True, max_length=512).input_ids.cuda()
            outputs_rank = my_experiment.model(input_ids=positive_index.cuda(), labels=my_experiment.index_token_positive[:len(q_id)])
            out_rank = outputs_rank["logits"][:,0,my_experiment.index_token_positive[0,0]]
            for index  in range(len(q_id)):
                q_id_c, d_id_c , rank = q_id[index].item(), d_id[index].item(), out_rank[index].item()
                if(q_id_c not in prediction_set):
                    prediction_set[q_id_c] = {}
                prediction_set[q_id_c][d_id_c] = rank
#getting the model
gt = dataset.qrels.get_dictionary()
gt = { str(q_id):{str(d_id): v for d_id, v in ds_id.items() } for q_id, ds_id in gt.items()}
pred ={ str(q_id):{str(d_id): v for d_id, v in ds_id.items() } for q_id, ds_id in prediction_set.items()}

import pytrec_eval

evaluator = pytrec_eval.RelevanceEvaluator(gt, {'map', 'ndcg', 'recip_rank.10'})
results = {}
evaluation =  evaluator.evaluate(pred)
for query, scores in evaluation.items():
    for score_key, score_val in scores.items():
        if(score_key not in results):
            results[score_key] = 0.
        results[score_key] += score_val
results = {k: v/len(evaluation) for k, v in results.items()}
print(results)