import dataloader
import showstuff
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

def gen_dataset(train_x, train_y, test_x, test_y):
    datasets = []
    for x, y in [(train_x, train_y), (test_x, test_y)]:
        x = torch.stack(
            [torch.Tensor(x[i]) for i in range(x.shape[0])]
        )
        y = torch.stack(
            [torch.Tensor(y[i:i+1]) for i in range(y.shape[0])]
        )
        datasets += [TensorDataset(x, y)]
        
    return datasets

train_dataset, test_dataset = gen_dataset(*dataloader.read_bci_data())

def runModels(
    models, epoch_size, batch_size, learning_rate, 
    optimizer = optim.Adam, criterion = nn.CrossEntropyLoss(),
    show = True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, len(test_dataset), shuffle=True)
    
    Accs = {
    **{key+"_train" : [] for key in models},
    **{key+"_test" : [] for key in models}
    }
    
    optimizers = {
        key: optimizer(value.parameters(), lr=learning_rate) 
        for key, value in models.items()
    }
    for epoch in range(epoch_size):
        train_correct = {key:0.0 for key in models}
        test_correct = {key:0.0 for key in models}
        # training multiple model
        for idx, data in enumerate(train_loader):
            x, y = data
            inputs = x.to(device)
            labels = y.to(device).long().view(-1)
        
            for optimizer in optimizers.values():
                optimizer.zero_grad()
        
            for key, model in models.items():
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            
                train_correct[key] += (
                    torch.max(outputs, 1)[1] == labels
                ).sum().item()
        
            for optimizer in optimizers.values():
                optimizer.step()
        
        # testing multiple model
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                x, y = data
                inputs = x.to(device)
                labels = y.to(device)
        
                for key, model in models.items():
                    outputs = model.forward(inputs)
        
                    test_correct[key] += (
                        torch.max(outputs, 1)[1] == labels.long().view(-1)
                    ).sum().item()

        for key, value in train_correct.items():
            Accs[key+"_train"] += [(value*100.0) / len(train_dataset)]
    
        for key, value in test_correct.items():
            Accs[key+"_test"] += [(value*100.0) / len(test_dataset)]
         
        if show:
            # clear_output(wait=True)
            showstuff.showAccuracy(
                title='Epoch [{:4d}]'.format(epoch + 1),
                **Accs
            )
        
    # epoch end
    torch.cuda.empty_cache()
    return Accs