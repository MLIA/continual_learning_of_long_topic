'''An experiment template for lire

'''
class LIReExperiment(object):
    def __init__(self, options, dataset):
        self.dataset = dataset
        self.options = options
        self.current_state = {}

    @property
    def state_dict(self):
        ''' Get metadata from experiments.
        get metadata of experiments object
        with important informations to continue
        experiments at this step. It also recursivelly call
        the state_dict property of contained object.
        '''
        state_dict = {}
        for k, v in self.__dict__.items():
            if(hasattr(v, "load_state_dict") and hasattr(v, 'state_dict')):
                state_dict[k] = v.state_dict()
            elif(hasattr(v, 'state_dict')):
                state_dict[k] = v.state_dict
        state_dict["options"] = self.options
        state_dict["experiment_state"] = self.current_state
        return state_dict

    @state_dict.setter
    def state_dict(self, value):
        ''' set metadata from experiments.
        set metadata of experiments object.
        Allow to load an experiment checkpoint
        '''
        for k, v in self.__dict__.items():
            if hasattr(v, "load_state_dict"):
                v.load_state_dict(value[k])
            elif hasattr(v, 'state_dict'):
                v.state_dict = value[k]

        self.options = value["options"]
        self.current_state = value["experiment_state"]

    def save_experiment(self, filepath):
        torch.save(self.state_dict, filepath)
    
    def __getitem__(self, key):
        return self.options.__dict__[key]
    
    def train(self):
        raise NotImplementedError
    
    def ending_task(self):
        raise NotImplementedError

    def begin_task(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError