
import logging
import torch
from torch import optim
import math
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from lire.experiment_tools.experiment_template import LIReExperiment
from lire.misc import struct
import time, datetime
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
default_options = struct({'save': '/local/gerald/default.pth',
                          'batch_size': 16,
                          'lr': 1e-3,
                          'beta1': 0.9,
                          'beta2': 0.999,
                          'device': 0,
                          'epoch_by_task': 1,
                          'loss_estimation_iter': 10000,
                          'number_iteration_by_task': None,
                          'model': 't5-base'})


class T5TaskRanker(LIReExperiment):
    def __init__(self, options, dataset):
        super(T5TaskRanker, self).__init__(options, dataset)
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s %(message)s')
        logging.info("Initialise tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained(options.model)
        logging.info("Loading pretrained model")
        self.model = T5ForConditionalGeneration.from_pretrained(options.model)
        self.model = self.model.to(self.options.device)

        logging.info("Use default optimizer Adam")
        self.optimiser = optim.Adam(self.model.parameters(),
                                    lr=self.options.lr,
                                    betas=(self.options.beta1,
                                           self.options.beta2))
        self.evaluation_task = []
        logging.info("Initialise meta info")
        self.training = True
        self.current_state = {'task': 0}
        token_positive = "true"
        token_negative = "false"
        index_token_positive =\
            self.tokenizer(token_positive, return_tensors="pt").input_ids
        index_token_negative =\
            self.tokenizer(token_negative, return_tensors="pt").input_ids
        self.index_token_positive =\
            index_token_positive.repeat(self.options.batch_size,
                                        1).to(options.device)
        self.index_token_negative =\
            index_token_negative.repeat(self.options.batch_size,
                                        1).to(options.device)
        self.state_dict_file = self.options.save

        def query_transform(x):
            return x + '</s>'

        def output_transformation(x):
            return (x[0] + x[1], x[0] + x[2])

        self.dataset.set_output_transformation(output_transformation)
        self.dataset.set_query_transform(query_transform)
        self.writer = SummaryWriter()
        self.current_state["min_loss"] = {}
        self.current_state["iteration"] = 0

    def step(self, positive, negative):
        with autocast(enabled=True):
            self.optimiser.zero_grad()
            positive_index, negative_index =\
                self.tokenizer(list(positive), return_tensors="pt",
                            padding=True, max_length=512).input_ids,\
                self.tokenizer(list(negative), return_tensors="pt",
                            padding=True, max_length=512).input_ids
            
            positive_index, negative_index =\
                positive_index.to(self.options.device),\
                negative_index.to(self.options.device)
            outputs_positive = self.model(input_ids=positive_index,
                                        labels=self.index_token_positive)
            outputs_negative = self.model(input_ids=negative_index,
                                        labels=self.index_token_negative)

            loss_positive = outputs_positive.loss
            loss_negative = outputs_negative.loss

            loss = loss_positive + loss_negative


        self.writer.add_scalar('train/batch_loss', loss.item(),
                            self.current_state["iteration"])
        return loss

    def begin_task(self):
        self.dataset.set_current_task_by_id(self.current_state['task'])
        self.dataloader =\
            data_utils.DataLoader(self.dataset,
                                  batch_size=self.options.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  num_workers=4)

    def ending_task(self):
        pass

    def _train_task(self, max_iteration, dataloader, shift=0):
        logging.info("Number of examples for the current task : " +
                     str(len(self.dataset)))
        if(self.current_state["task"] not in self.current_state["min_loss"]):
            self.current_state["min_loss"][self.current_state["task"]] =\
                math.inf
        start_time = time.time()
        for i in range(max_iteration):
            loss_sum = 0.
            loss_n = 0
            for it, (pos, neg) in enumerate(self.dataloader):

                batch_loss = self.step(pos, neg)
                
                self.scaler.scale(batch_loss).backward()
                self.scaler.step(self.optimiser)
                self.scaler.update()
                loss_sum += batch_loss.item() * len(pos)
                loss_n += len(pos)

                self.current_state["iteration"] += 1

                if(it % self.options.loss_estimation_iter == 10):
                    task_iteration = i*len(self.dataset) + it * self.options.batch_size
                    estimated_loss = loss_sum/loss_n
                    self.writer.add_scalar('train/avg_loss',
                                           estimated_loss,
                                           self.current_state["iteration"])
                    self.evaluate()
                    if(self.current_state["min_loss"]
                       [self.current_state["task"]] > estimated_loss):
                        torch.save(self.state_dict, self.state_dict_file)
                        self.current_state["min_loss"][self.current_state["task"]] = estimated_loss
                    
                    if(self.options.number_iteration_by_task is not None):
                        if(callable(self.options.number_iteration_by_task )):
                            nb_it_task = self.options.number_iteration_by_task()
                        else:
                            nb_it_task = self.options.number_iteration_by_task
                        EAT = (time.time() - start_time)/task_iteration *\
                            (nb_it_task - task_iteration)
                        logging.info("Loss for iteration " + str(task_iteration) +
                                     " is " + str(estimated_loss) +
                                     " EAT: " +str(datetime.timedelta(seconds=EAT)))
                        if(nb_it_task <=
                           task_iteration):
                            return
                    else:
                        logging.info("Loss for iteration " + str(task_iteration) +
                                 " is " + str(estimated_loss))

    def add_evaluation(self, function, parameters):
        self.evaluation_task.append((function, parameters))

    def evaluate(self):
        for k, v in self.evaluation_task:
            k(*v)

    def run(self):
        self.scaler = GradScaler(enabled=True)
        if(self.training):
            epoch_by_task = self.options.epoch_by_task
            for i in range(self.dataset.get_nb_tasks()):
                self.current_state['task'] = i

                self.begin_task()
                self._train_task(epoch_by_task, self.dataloader)
                self.ending_task()
                self.optimiser = optim.Adam(self.model.parameters(),
                                            lr=self.options.lr,
                                            betas=(self.options.beta1,
                                                   self.options.beta2))