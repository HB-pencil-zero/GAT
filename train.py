from model import *
from layers import *
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import argparse
import numpy as np 
from utils import *

parser=argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--headersNum',type=int, default =3 ,      
                    help="headers num on attention mechanism"    )

args=parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GAT(n_features=features.shape[1],
            n_hidden= args.hidden,
            n_class=labels.max().item() + 1,
            headerNums=args.headersNum
            )


optimizer = optim.Adam(model.parameters(recurse=True),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = torch.from_numpy(np.array(features)).cuda()
    adj = torch.from_numpy(np.array(adj)).cuda()
    labels = torch.from_numpy(np.array(labels)).cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch:int,criterion=CrossEntropyLoss()):
    model.train()
    model.double()
    optimizer.zero_grad()
    output=model(features.double().cuda(),adj.double().cuda())
    loss=criterion(output[idx_train],labels[idx_train].long())
    loss.backward()
    print(model.layerlist[0].weightlist[0].grad)
    optimizer.step()
    accu=accuracy(output[idx_train],labels[idx_train].long())
    print("epoch {} train finished , and accuracy is {}".format(epoch+1,accu))


def test(epoch:int):
    model.eval()
    with torch.no_grad():
        output=model(features.double(),adj.double())
        loss=CrossEntropyLoss()(output[idx_val],labels[idx_val].long())
        accu=accuracy(output[idx_val],labels[idx_val].long())
        print("epoch {} test, and accuracy is {}".format(epoch+1,accu))

if __name__ == '__main__':
    epochs=args.epochs
    for i in range(epochs):
        train(epoch=i)
        test(epoch=i)
        print(''.join(list("*" for i in range(10))))
    print("finished all epochs train")
    
