# ContinualPassageRanking

Repository for the paper "Continual Learning of Long Topic Sequences in Neural Information Retrieval". In this repository you will find the code for generating the long sequences topics/tasks. 

## Dependency
pytorch : 
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`

Transformers and pyserini
` pip install transformers pyserini numpy`
## Dataset
Dataset can be generated based on their topics using a sentence bert pretrained model. We also provide the three set used in the associated paper.


## Training model
In the following we describe the different approaches tested in successively fine-tunning model.
