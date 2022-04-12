## Experiment template

The class **LIReExperiment** provide the template for experiments (call by ray tune for random/grid-search). It provide the structure such as begin_task/ending_task or run template. Most of all, it allow to automatically get the state dict from contained objects and load them.