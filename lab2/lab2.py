import dataloader
import show
import run
import AccuracyResult
import EEGNet
import DeepConvNet
import torch
import torch.nn as nn

if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print('pytorch device : ', device)

    AccRes = AccuracyResult.AccuracyResult()

    print('Training and Testing EEGNet')
    models = {
        "elu" : EEGNet.Net(nn.ELU).to(device),
        "relu" : EEGNet.Net(nn.ReLU).to(device),
        "leaky_relu" : EEGNet.Net(nn.LeakyReLU).to(device),
    }
    Accs = run.runModels(models, epoch_size=300, batch_size=64, learning_rate=1e-2, show=False)
    show.showAccuracy("EEGNet", **Accs)
    AccRes.add("EEGNet", Accs, show=False)

    print('Training & Testing DeepConvNet')
    models = {
        "elu" : DeepConvNet.Net(nn.ELU).to(device),
        "relu" : DeepConvNet.Net(nn.ReLU).to(device),
        "leaky_relu" : DeepConvNet.Net(nn.LeakyReLU).to(device),
    }
    Accs = run.runModels(models, epoch_size=300, batch_size=64, learning_rate=1e-3, show=False)
    show.showAccuracy("DeepConvNet", **Accs)
    AccRes.add("DeepConvNet", Accs, show=False)
    
    AccRes.show()