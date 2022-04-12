
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn 
from torch import optim
from lire.function_tools import gradient_tools
from lire.continual_models import ewc
# Loading MNIST
dataset = torchvision.datasets.MNIST

training_data = dataset("./data", train=True, download=True)
testing_data = dataset("./data", train=False, download=True)


def sub_dataset(dataset, classes):
    new_dataset = []
    for x, y  in dataset:
        if(y in classes):
            new_dataset.append((x,y))
    return new_dataset

def flat_tranform(x):
    return x.view(-1)

# Loading MNIST
dataset = torchvision.datasets.MNIST

training_data = dataset("./data", train=True)
testing_data = dataset("./data", train=False)

data_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5,), (0.5,)),
                               flat_tranform]
)


def sub_dataset(dataset, classes):
    new_dataset = []
    X, Y = [], []
    classes = classes.tolist()
    print(classes)
    for x, y  in dataset:
        if(y in classes):
            X.append(data_transform(x))
            Y.append(torch.LongTensor([y]))
        if(len(X) >= 2000):
            return X, Y
    return X, Y

def flat_tranform(x):
    return x.view(-1)

print("Building tasks")
sub_classes = torch.randperm(10)
sub_classes = [sub_classes[0:2], sub_classes[2:4], sub_classes[4:6], sub_classes[6:8], sub_classes[8:]]
tasks = [sub_dataset(training_data, task_class) for task_class in sub_classes]

print("Instanciate model")
encoder = nn.Linear(784, 20)
decoder = nn.Linear(20, 10)
model = nn.Sequential(encoder, nn.ReLU(), decoder, nn.ReLU())
criterion = nn.CrossEntropyLoss()

class LogProbModel(nn.Module):
    def __init__(self, model, criterion, parameters_list):
        super(LogProbModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.parameters_list = parameters_list
    
    def forward(self, data):
        data
        y_pred = self.model(x.unsqueeze(0))
        log_prob = self.criterion(y_pred, y)
        return log_prob
    
    def parameters(self): 
        return self.parameters_list

print("Getting weights")
weights = list(model.parameters())
print("w shape ", [i.size() for  i in weights])

optimizer = optim.SGD(weights, lr=1e-1)
# log_prob_model = LogProbModel(model, criterion, weights)
# ewc_model = ewc.EWC(log_prob_model)

class SDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return len(self.X)


for task_index, task_data in enumerate(tasks):
    # if(task_index > 0):
    #     ewc_model.next_task(task_data[0], task_data[1], log_prob_model)
    dataloader = torch.utils.data.DataLoader(SDataset(task_data[0], task_data[1]), batch_size=50, shuffle=True, num_workers=2)
    for _ in range(50):

        for j in range(0, task_index+1):
            with torch.no_grad():
                dataloader = torch.utils.data.DataLoader(SDataset(tasks[j][0], tasks[j][1]), batch_size=100, shuffle=False, num_workers=2)
                total_classifier  = 0
                for x, y in dataloader:
                    y_pred = model(x)
                    # print(x.shape, y.shape)
                    _, index = y_pred.max(-1)
                    # print(index.shape)
                    total_classifier += (index == y.squeeze()).sum()
                print(j, '(',sub_classes[j],')','->', total_classifier/len(dataloader.dataset))

        for x, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y.squeeze())
            # if(task_index > 0):
            #     loss += 0. *  ewc_model.ewc_loss()
            loss.backward()
            optimizer.step()


    print("------")


# # EWC
# # 0 ( [0, 1] ) -> tensor(0.9912)
# # 1 ( [2, 3] ) -> tensor(0.8789)
# # 2 ( [4, 5] ) -> tensor(0.9730)
# # 3 ( [6, 7] ) -> tensor(0.9881)
# # 4 ( [8, 9] ) -> tensor(0.9730)

# # No EWC
# # 0 ( [0, 1] ) -> tensor(0.8654)
# # 1 ( [2, 3] ) -> tensor(0.9361)
# # 2 ( [4, 5] ) -> tensor(0.9747)
# # 3 ( [6, 7] ) -> tensor(0.9888)
# # 4 ( [8, 9] ) -> tensor(0.9752)