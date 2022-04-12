from torch.nn import Module
from torch.distributions import Distribution
from collections.abc import Iterable


def estimate_fisher_diag(input_iterator: Iterable,
                         parametric_function: Module,
                         expectation_estimation_sample: int = 1) -> Iterable:

    parameters = list(parametric_function.parameters())
    fisher_diag = []
    for w in parameters:
        # Initialise and ensure their is no gradient on parameters
        fisher_diag.append(w.new_zeros(w.size()))
        w.grad.data.zero_()

    for _ in range(expectation_estimation_sample):
        for data in input_iterator:
            if isinstance(data, Distribution):
                parametric_function(data).backward()
            else:
                parametric_function(data).backward()

            for f, w in zip(fisher_diag, parameters):
                f.add_(w.grad.data.pow(2))
            parametric_function.zero_grad()
    for f in fisher_diag:
        f.div_(expectation_estimation_sample * len(input_iterator))

    return fisher_diag
