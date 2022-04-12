from torch.utils.data import Dataset
from torch.distributions.distribution import Distribution

class ContinualDataset(Dataset):
    def __init__(self):
        super(ContinualDataset, self).__init__()
        self._task_id = 0

    def set_current_task_by_id(self, task_id):
        self._task_id = task_id
    
    def get_current_task_id(self):
        return self._task_id

    @property
    def state_dict(self):
        ''' Get metadata from experiments.
        get metadata of experiments object
        with important informations to continue
        experiments at this step. 
        '''
        return {"task_id": self.get_current_task_id()}

    @state_dict.setter
    def state_dict(self, value):
        ''' set metadata from experiments.
        set metadata of experiments object.
        Allow to load an experiment checkpoint
        '''
        self.set_current_task_by_id(value["task_id"])

    def get_nb_tasks(self):
        raise NotImplementedError

class ContinualDatasetSplitter(ContinualDataset):
    def __init__(self, dataset,
                 fold={'train': 0.8, 'val':0.1, 'test':0.1},
                 n_tasks=5, size=[0.1,0.3,0.4,0.1,0.1], seed=42):
        super(ContinualDatasetSplitter, self).__init__()
        self._n_tasks = n_tasks 
        self._fold = fold
        self._size = size
        self._dataset = dataset
        self._seed = seed
        self._tasks = None
        self._current_fold = list(fold.keys())[0]
        assert(issubclass(size.__class__, Distribution) or n_tasks == len(size))
        assert(sum(fold.values(), 0) == 1)
        self._prepare()

    def _get_task_size(self, task_index):
        if hasattr(self.size, "__getitem__"):
            return self.size[task_index]


    def _prepare(self):
        dataset_size = len(self._dataset)
        torch.manual_seed(self._seed)
        random_permutation = torch.randperm(dataset_size)
        tasks = list() 
        head_index = 0
        for i in range(self._n_tasks):
            task_size = self._get_task_size(i)
            task_indexes = random_permutation[head_index: head_index + task_size]

            task_fold = {}
            head_fold = 0
            for k,v in self.fold.items():
                task_fold[k] = task_indexes[head_fold:int(head_fold + v*task_size)]
            
            tasks.append(task_fold)
        self._tasks = task


    # def set_current_fold(self, fold):
    #     assert(fold is in self.fold)
    
    # def get_current_fold(self):
    #     self._getter = lambda x: self.
    #     return self._current_fold

    def __len__(self):
        return len(self._tasks[self._current_fold][self.get_current_task_id()])
    
    def __getitem__(self, index):
        return self._dataset[self._tasks[self._current_fold][self.get_current_task_id()][index]]

    def get_nb_tasks(self):
        return len(self._tasks)
    
    def save_tasks(self, folder_path):
        pass

class ContinualDatasetTaskCat(ContinualDataset):
    def __init__(self, continual_dataset_list,
                 intra_order_rule="random",
                 extra_order_rule="random"):
        super(ContinualDatasetTaskCat, self).__init__()
        self._datasets = continual_dataset_list
        self._intra_order = intra_order_rule
        self._extra_order = extra_order_rule
        self._tasks = []
    
    def _prepare(self):
        self._meta_task_indexes = {i * len(dataset) + j: (i, j) 
                                    for i, d in  enumerate(self._datasets) 
                                    for j in range(d.get_nb_tasks())}
        if(self._intra_order == "random" and self._extra_order == "random"):
            self.task_indexes = torch.randperm(len(self._meta_task_indexes))

    def get_nb_tasks(self):
        return len(self._tasks)

    def __len__(self):
        task_id = self.get_current_task_id()
        dataset_index, sub_task_index = self._meta_task_indexes[task_id]
        self.current_dataset = self._datasets[dataset_index]
        self.current_dataset.set_current_task_by_id(sub_task_index)

        return len(self.current_dataset)
    
    def __getitem__(self, index):
        task_id = self.get_current_task_id()
        dataset_index, sub_task_index = self._meta_task_indexes[task_id]
        self.current_dataset = self._datasets[dataset_index]
        self.current_dataset.set_current_task_by_id(sub_task_index)

        return self.current_dataset[index]
    