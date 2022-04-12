from lire.function_tools import gradient_tools

class EWC(object):
    def __init__(self, model_shared):
        self.model = model_shared
        self.weights = list(self.model.parameters())
        self.fisher = [w.new_zeros(w.shape).data for w in  self.weights]

    def next_task(self, data, log_prob):
        self.star_weights = [w.data.clone() for w in  self.weights]
        self.local_fisher = gradient_tools.estimate_fisher_diag(data, log_prob)
        self.fisher = [f + lf for f, lf in  zip(self.fisher, self.local_fisher)]

    def ewc_loss(self, alpha=1):
        ewc_loss_backward = 0.
        for f, w, sw in zip(self.fisher, self.weights, self.star_weights):
            ewc_loss_backward += (f.detach() * (w - sw.detach())**2).sum()
        return ewc_loss_backward
