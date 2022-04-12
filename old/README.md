# Lifelong Information Retrieval

A Lifelong Learning pytorch package for information retrieval. The model are based on pytorch libraries using in addition [Hugging Face](https://huggingface.co/models) models. This library named **L**ifelong **I**nformation **Re**trieval (**LIRe**) provide tools and implement the proposale of the paper.

## Install and configure
We use conda to configure if anaconda is not installed run 

To create the envs lire with our configuration please run :
```bash
$ bash configure.sh
```
To activate the environment source the activate.sh file :
```bash
$ source activate.sh 
```
## The continual graph

As experiments can make time to train, we propose a simple framework for sequences. At each new tasks the package save the models, when learning severall times a same start sequence we do not want to re-train model for each tasks. Thus we propose to make a sequence graph, branching a new subgraph when encounter an unknow "next tasks". 
Let consider a sequence as a set of corpus, then we can construct the graph sequence using the factory graph function *lire.setting_tools.continual_task_graph.factory_task_graph(iterable dataset_id_list) -> ContinualTaskGraph*.

``` python
from lire.setting_tools import continual_task_graph as ctg

dataset_id_list = [
    ['msmarco_sub_1', 'ms_marco_sub_2', 'ms_marco_sub_3', 'ms_marco_sub_4'],
    ['msmarco_sub_1', 'ms_marco_sub_2', 'ms_marco_sub_4', 'ms_marco_sub_3'],
    ['msmarco_sub_1', 'ms_marco_sub_3', 'ms_marco_sub_2', 'ms_marco_sub_4'],
    ['msmarco_sub_3', 'ms_marco_sub_1', 'ms_marco_sub_2', 'ms_marco_sub_4']
    ]
my_continual_graph = ctg.factory_task_graph(dataset_id_list)
```

Therefore adding new sequence to the graph can be added if a start sequence exists all model will be not re-trained if precised in training paremeters (see training using continual task graph). 
Once all modules of the graph learned and saved, one can use sequence visualisation tools to efficiently render performances.


## The model
All continual methods must inherits from class **ContinualModelisation** which extend nn.Module pytorch class. Those continual methods can be model, regularisation or memory oriented.

see [specific readme](lire/continual_models/readme.md)

