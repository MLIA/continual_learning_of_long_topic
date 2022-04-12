# Datasets

The dataset is dcomposed as follow :
* A class loading the dataset
* Configuration files in **dataset_info** folder containing url to download and others additional information

## The IRClass

## The qrels format
The qrels format come from the trec evaluation, which inform the relevance of a query or a topic with a document. The qrels format is designed as following:

```
1 0 AP880212-0161 0
1 0 AP880216-0139 1
1 0 AP880216-0169 0
1 0 AP880217-0026 0
1 0 AP880217-0030 0
```
with the first information being the topic/query id the second the feedback iteration, the third the document and the last the relevance being 1 for relevant query/document  and 0 for irelevant. We sum up the format in the table below.

| Topic/Query ID | Feedback Iteration | Document ID | Binary relevance |
| -------------- | ------------------ | ----------- | ---------------- |
|      523       |         0          | D957846     |  1               | 
|      523       |         0          | D957847     |  0               | 
|      749       |         0          | D957846     |  1               | 


Loading qrel files can be done on memory contrary to documents. We propose a class named **Qrels** allowing to load and get relevant line/columns. To load data you can use the following code:

```python
from lire.data_tools import data_reader
qrels = data_reader.Qrels(irrelevant=True,
                          use_inverse_dictionary=True, 
                          getter_expression="qi -> q, dr, dn")
qrels.set_ressource(path_to_qrels_file)
```
The getter expression define what is the input/output of the **__getitem__** methods, with for input **qi** the index of the query and for the output **q** the query original index, **dr** relevant documents index and **dn** not relevant documents index.

For instance given the following code instruction (MSmarco train_qrels) :
> qrels[169]

will give the following output:
> ('670628', {'25886', '25887'}, set())

being a tuple of query index, the set of relevant documents and a set of not relevant documents (in this case none are given). 

## The microblog dataset and Twitter API

You can access to data  at (https://twitter.com/anybody/status/311733352694685696)

## Loading huge dataset
Memory usage can lead to full all memory to avoid it dataset can be splitted into severall file.

## The MSMarco Datasets
The MSMarco datasets are available in lire.data_tools.dataset.MSMarco. Different configuration exists, the document ranking (**MSMarcoDocumentRanking**) and the passage ranking used in the experiments (**MSMarcoPassageRanking...**).
### MSMarcoPassageRanking
The splits available for MSMarco passage ranking are "train", "dev" and "eval", however we do not have at the user's disposal qrels file associated to the evaluation set ("eval" split).
As the document ranking we can get the triplet training file given on the [repository](https://github.com/microsoft/MSMARCO-Passage-Ranking). To create the dataset you should instanciate the dataset as following:
```python 
from lire.data_tools.dataset import MSMarco
dataset =\
    MSMarco.MSMarcoPassageRankingDataset("~/data",
                                         download=True,
                                         split="train",
                                         getter='triplet')
```
Identically, we can choose between differents getter function (used calling operator *__getitem__*). The "positive" getter return all positive documents for a query, the example below show the mechanism :
```python 
from lire.data_tools.dataset import MSMarco
dataset =\
    MSMarco.MSMarcoPassageRankingDataset("~/data",
                                         download=True,
                                         split="dev",
                                         getter='positive')
print('sample of the dataset : ',dataset[1])
```
which will output the following tuple of query and list of relevant documents : 
> $ sample of the dataset :  ('why did rachel carson write an obligation to endure', ['Carson believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment with, for example, DDT and harming living things. Carson adds that the intensification of agriculture is causing other major problems, like newly developed or created insects and diseases.', "The Obligation to Endure by Rachel Carson Rachel Carson's essay on The Obligation to Endure, is a very convincing argument about the harmful uses of chemicals, pesticides, herbicides, and fertilizers on the environment."])


To adapt to lifelong learning we propose split of the dataset based on the sentence embeddings using different models. To this end we selected pretrained embedding of sentences similarity model based on the **sentence-transformers** [library](https://www.sbert.net/index.html). The script used to make those datasets are available [here](../../../examples/question_clustering.ipynb)


An example below of topics on the *dev* set: 

Topic 0 : 
* how many grams of protein in pork chops
* are there different grades of pork
* what temp and how long do you smoke 1 pork steaks in a 

Topic 2 :
* how long should it take for a fridge to cool
* what is freezing celsius temperature
* what is blue ice in a lake

Topic 4 :
* cost of aflac dental insurance
* compare cost of dental implants
* what is takes to be a dentist

Topic 564 :
* what is aromatherapy menu at hotel?
* does fish oil help pain

The default dataset **MSMarcoPassageRankingDatasetTopicCleanedQueryDataset** contains **3402** topics with at least 3 samples in the *dev* subsets and *8* samples in the *train* subsets.
This dataset inherit from **MSMarcoPassageRankingDataset** bu also from **ContinualDataset**([source](../continual_routine.py)) thus changing topics is made by continual object methods.

You can sample the corpus using the following [notebook](example/reading_msmarco_ir_topics.ipynb).


## The TREC-COVID datasets


## The AOL datasets