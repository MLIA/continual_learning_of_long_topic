'''The T5 ranking baseline.
The baseline implemented here rely on the paper
"Document Ranking with a Pretrained Sequence-to-Sequence Model"
published at EMNLP (2020).
'''
import torch
import logging
from lire.data_tools.dataset import MSMarco
import tqdm
import torch.utils.data as data_utils
import pytrec_eval

from lire.misc import struct
import time, datetime
from torch.cuda.amp import autocast
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
dataset.set_current_task_by_id = lambda x : 1 
def fnull():
    return 1
dataset.get_nb_tasks = fnull

logging.info('Configure the experiment')

nb_queries = dataset.get_nb_annotated_queries()
options = t5ranker.default_options
options.batch_size = 4
options.device = 1
batch_size = options.batch_size
nb_iteration = nb_queries * 9
options.number_iteration_by_task = int(nb_iteration)
options.save = '/local/gerald/T5RankerBaselineMSMarco-25-02-2021.pth'
print(options)
logging.info("Number of iteration considering the batch size: " +
             str(nb_iteration))
my_experiment = t5ranker.T5TaskRanker(options, dataset)
# logging.info("Loading check point")
# state_dict = torch.load('/local/gerald/T5RankerBaselineMSMarco.pth')
# my_experiment.state_dict = state_dict
# my_experiment.options.batch_size = 4
# logging.info("Continue experiment")

# an eval set

dataset_eval = MSMarco.MSMarcoPassageRankingDataset(data_folder,
                                               download=False,
                                               split="dev",
                                               getter='top1000', 
                                               subset_size=30)
dataset_eval.set_output_transformation(lambda x : x[0] + x[1])
dataloader_eval =\
    data_utils.DataLoader(dataset_eval,
                            batch_size=32,
                            drop_last=False,
                            shuffle=False,
                            num_workers=4)

gt = dataset_eval.qrels.get_dictionary()
def evaluation_method(xp, dataloader, evaluator):
    prediction_set = {}
    with torch.no_grad():
        token_positive = "true"
        index_token_positive =\
                xp.tokenizer(token_positive, return_tensors="pt").input_ids
        index_token_positive =\
            index_token_positive.repeat(32,
                                        1).to(xp.options.device)

        with autocast(enabled=True):
            for q_id, d_id, input_tokens in  dataloader:
                positive_index =\
                    xp.tokenizer(input_tokens, return_tensors="pt", padding=True, max_length=512).input_ids.to(xp.options.device)
                outputs_rank = xp.model(input_ids=positive_index, labels=index_token_positive[:len(q_id)])
                out_rank = outputs_rank["logits"][:,0,index_token_positive[0,0]]
                for index  in range(len(q_id)):
                    q_id_c, d_id_c , rank = q_id[index].item(), d_id[index].item(), out_rank[index].item()
                    if(q_id_c not in prediction_set):
                        prediction_set[q_id_c] = {}
                    prediction_set[q_id_c][d_id_c] = rank
    pred ={ str(q_id):{str(d_id): v for d_id, v in ds_id.items() } for q_id, ds_id in prediction_set.items()}
    results = {}
    evaluation =  evaluator.evaluate(pred)
    print(evaluation)
    for query, scores in evaluation.items():
        for score_key, score_val in scores.items():
            if(score_key not in results):
                results[score_key] = 0.
            results[score_key] += score_val
    results = {k: v/len(evaluation) for k, v in results.items()}
    print(results)
    xp.writer.add_scalar('dev/mrr',
                         results['recip_rank'],
                         xp.current_state["iteration"])
    xp.writer.add_scalar('dev/map',
                         results['map'],
                         xp.current_state["iteration"])
    xp.writer.add_scalar('dev/ndcg',
                         results['ndcg'],
                         xp.current_state["iteration"])
gt = { str(q_id):{str(d_id): v for d_id, v in ds_id.items() } for q_id, ds_id in gt.items()}
evaluator = pytrec_eval.RelevanceEvaluator(gt, {'map', 'ndcg', 'recip_rank.10'})
my_experiment.add_evaluation(evaluation_method, [my_experiment, dataloader_eval, evaluator])
my_experiment.run()
