# Experiments

## T5 ranker baseline
The script associated to this experiment is **"t5_ranker_baseline.py"**. The experiment reproduce the paper "[Document Ranking with a Pretrained Sequence-to-Sequence Model](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_etal_FindingsEMNLP2020.pdf)", it make part of the strong baseline where all tasks are mixed togethers. Some points however differs, for instance the queries seen is not exactly the same but the number of training samples is selected to be on the same order (10 epochs, with epochs corresponding to annotated queries in the qrels file).
Waiting for refactor, the experiment will output a file "/local/gerald/T5RankerBaselineMSMarco.pth" contaning the state dict (see [experiment_tools doc](../lire/experiment_tools/readme.md)) and will log into tensorboard runs folder as default.
Information are given in the table below:

| Infos         | values        | Config        |
| ------------- |:-------------:|:-------------:|
| Training time, batch 16, small | ~ 40 hours    |1 X 1070    |
| Training time, batch 32, small | ~ 17 hours    |1 X 2080TI    |
| Training time, batch 32, base, (16bits)  | ~ 17 hours    |1 X 2080TI    |
