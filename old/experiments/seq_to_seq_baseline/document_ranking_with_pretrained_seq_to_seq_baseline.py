import argparse
import logging
from os.path import join
import os
import time
import tqdm
import torch
import torch.utils.data as data_utils
from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
import random
from ray import tune
from transformers import T5Tokenizer, T5TokenizerFast
from transformers import T5ForConditionalGeneration

from lire.data_tools.dataset import MSMarco

parser = argparse.ArgumentParser(description='Configuration of the experiment Based on the model proposed in "Document Ranking with Pretrained Sequence to Sequence model"')

# Mandatory parameters

parser.add_argument('--data-folder', type=str, dest='data_folder',
                    help='The folder where data are or will be downloaded/decompressed')
parser.add_argument('--model-storage-folder', type=str, dest='model_storage_folder',
                    help='folder to save the model')
# Optional parameters (with default values but important for model performances/training time)

parser.add_argument('--batch-size', type=int, dest='batch_size', default=10,
                    help='The size of the batch')
parser.add_argument('--max-iter', type=int, dest='max_iter', default=5000,
                    help='Maximum number of iteration by epoch')
parser.add_argument('--max-epoch', type=int, dest='max_epoch', default=1,
                    help='Maximum number of epoch')
parser.add_argument('--devices', type=int, dest='devices', default=-1,
                    help='Device to use (GPU) if -1 using cpu')
parser.add_argument('--subset-size', type=int, dest='subset_size', default=None,
                    help='Considering only a subset of MSMarco (for dev or testing)')

# Less important parameters (with default value not impacting learning)

parser.add_argument('--loss-estimation-iter', type=int, dest='loss_estimation_iter', default=500,
                    help='Number of iteration to average and print the loss')

args = parser.parse_args()


logging.info("Instanciate the model")

def training_function(config):
    args = type('new_dict', (object,), config)
    key_time = round(time.time() * 1000)
    os.makedirs(args.model_storage_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    logging.info("Configuring lambda functions")


    logging.info("Loading MSMarco Training set")

    data_folder = args.data_folder
    split = 'train'
    dataset = MSMarco.MSMarcoPassageRankingDataset(data_folder, download=True, split=split, storage='full', getter='triplet', subset_size=args.subset_size)

    logging.info("Configuring dataset and pipeline")

    set_device = lambda x : x if(args.devices == -1) else x.to(args.devices)

    identity_function = lambda x : x

    tokenizer = T5TokenizerFast.from_pretrained("t5-small")

    query_transform = lambda x: x+'</s>'

    output_transformation = lambda x: (x[0] + x[1], x[0] + x[2])
    dataset.set_output_transformation(output_transformation)
    dataset.set_query_transform(query_transform)
    training_dataloader = data_utils.DataLoader(dataset, batch_size=args.batch_size, drop_last=True)

    logging.info("Configuring label set")

    token_positive = "true"
    token_negative = "false"
    index_token_positive = tokenizer(token_positive,return_tensors="pt").input_ids
    index_token_negative = tokenizer(token_negative, return_tensors="pt").input_ids
    index_token_positive = set_device(index_token_positive.repeat(args.batch_size, 1))
    index_token_negative = set_device(index_token_negative.repeat(args.batch_size, 1))


    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model = set_device(model)
    adam_optimizer = optim.Adam(model.parameters())

    logging.info("Start fine tunning")

    scaler = GradScaler(enabled=True)
    loss_accumulator = 0.
    progress_bar = tqdm.trange(len(training_dataloader))
    min_loss = 10000.
    for epoch in range(args.max_epoch):
        for it, (positive, negative) in zip(progress_bar, training_dataloader):
            adam_optimizer.zero_grad()
            with autocast(enabled=True):
                positive_index, negative_index =\
                    tokenizer(list(positive), return_tensors="pt", padding=True, max_length=512).input_ids,\
                    tokenizer(list(negative), return_tensors="pt", padding=True, max_length=512).input_ids
                positive_index, negative_index = set_device(positive_index), set_device(negative_index)

                outputs_positive = model(input_ids=positive_index, labels=index_token_positive)
                outputs_negative = model(input_ids=negative_index, labels=index_token_negative)
                loss_positive = outputs_positive.loss
                loss_negative = outputs_negative.loss
            
                loss = loss_positive + loss_negative

                loss_accumulator = loss.item()
                
                if(it%args.loss_estimation_iter == 0 and it != 0):
                    progress_bar.set_postfix({"L":loss_accumulator/args.loss_estimation_iter })
                    if(min_loss > loss_accumulator/args.loss_estimation_iter):
                        
                        torch.save(model.state_dict(), join(args.model_storage_folder, str(key_time)+".data"))
                        min_loss = loss_accumulator/args.loss_estimation_iter
                    tune.report(mean_loss=loss_accumulator/args.loss_estimation_iter)
                    loss_accumulator = 0.


            scaler.scale(loss).backward()
            
            scaler.step(adam_optimizer)
            scaler.update()
            if(it == args.max_iter):
                break








config = args.__dict__
config["max_iter"] =  tune.grid_search([5001, 10001, 15001])
analysis = tune.run(training_function, config=config,  resources_per_trial={"gpu": 1})

print("Best config: ", analysis.get_best_config(
    metric="mean_loss", mode="min"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df