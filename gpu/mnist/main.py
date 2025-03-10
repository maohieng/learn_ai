from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import os
import time
from main_resnet import SimpleResNet


class Net(nn.Module):
    def __init__(self, disable_dropout=False):
        super(Net, self).__init__()
        self.disable_dropout = disable_dropout
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if not self.disable_dropout:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if not self.disable_dropout:
            x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_losses = 0
    started = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        loss_item = loss.item()
        train_losses += loss_item
        if args.verbos and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_item))
    
    duration = time.time() - started
    return train_losses / len(train_loader.dataset), duration


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    started = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    duration = time.time() - started
    
    if args.verbos:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    return test_loss, duration

def get_suffix_filename(args, device):
    return f'{args.epochs}_{args.batch_size}_{args.lr}_{args.gamma}_{device}'

def get_export_dir(args):
    exp_dir = f'exports/lr{args.lr}/bs{args.batch_size}/gm{args.gamma}'
    
    if args.model_name != 'cnn':
        exp_dir += f'/{args.model_name}'

    print(f'Storing in {exp_dir} folder')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    return exp_dir

def plot_losses(exp_dir, suffix_filename, losses, disable_dropout = False, reverse_data = False):
    import matplotlib.pyplot as plt
    import pandas as pd

    title = 'Train and Test Losses'
    if disable_dropout:
        title += ' (Dropout Disabled)'
    if reverse_data:
        title += ' (Reverse Data)'

    plt.plot(losses['train_loss'], label='Train loss')
    plt.plot(losses['test_loss'], label='Test loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(f'{exp_dir}/losses_{suffix_filename}.png')
    # plt.show()

def plot_all_losses(exp_dir, suffix_filename):
    import matplotlib.pyplot as plt
    import pandas as pd

    title = 'Train Losses'

    # Plot 2 losses graph with epoch as legend
    losses1_file = f"{exp_dir}/losses_{suffix_filename}.csv"
    print(f'Loading {losses1_file}')
    losses1 = pd.read_csv(losses1_file)
    
    losses2_file = f'{exp_dir}/losses_{suffix_filename}_reverse_data.csv'
    print(f'Loading {losses2_file}')
    losses2 = pd.read_csv(losses2_file)

    losses3_file = f'{exp_dir}/losses_{suffix_filename}_dropout_disabled.csv'
    print(f'Loading {losses3_file}')
    losses3 = pd.read_csv(losses3_file)

    losses4_file = f'{exp_dir}/losses_{suffix_filename}_dropout_disabled_reverse_data.csv'
    print(f'Loading {losses4_file}')
    losses4 = pd.read_csv(losses4_file)

    # Plot big graph
    plt.figure(figsize=(10, 10))

    # Plot 4 graphs
    plt.plot(losses1['train_loss'], label='Train Loss', color='red')
    plt.plot(losses2['train_loss'], label='Train Loss (Reverse Data)', color='blue')
    plt.plot(losses4['train_loss'], label='Train Loss (Dropout Disabled, Reverse Data)' , color='purple')
    plt.plot(losses3['train_loss'], label='Train Loss (Dropout Disabled)' , color='green')
    
    # Limit y axis from 0 to 1
    # plt.ylim(0, 0.5)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title(title)
    img_file = f'{exp_dir}/losses_{suffix_filename}_merged_train.png'
    print(f'Saving {img_file}')
    plt.savefig(img_file)
    # plt.show()

    # Plot and save test losses
    title = 'Test Losses'
    plt.figure(figsize=(10, 10))

    # Plot 4 graphs
    plt.plot(losses1['test_loss'], label='Test Loss', color='red')
    plt.plot(losses2['test_loss'], label='Test Loss (Reverse Data)', color='blue')
    plt.plot(losses4['test_loss'], label='Test Loss (Dropout Disabled, Reverse Data)', color='purple')
    plt.plot(losses3['test_loss'], label='Test Loss (Dropout Disabled)', color='green')

    # Limit y axis from 0 to 1
    plt.ylim(0, 0.5)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(title)
    img_file = f'{exp_dir}/losses_{suffix_filename}_merged_test.png'
    print(f'Saving {img_file}')
    plt.savefig(img_file)
    # plt.show()

def print_time(trained_time_consumed, test_time_consumed):
    print(f"Total training time: {sum(trained_time_consumed)}, Average test time: {sum(test_time_consumed) / len(test_time_consumed)}")

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--read-only', action='store_true', default=False,
                        help='For reading losses and plot')
    parser.add_argument('--reverse-data', action='store_true', default=False,
                        help='For data reversing train for test and test for train')
    parser.add_argument('--disable-dropout', action='store_true', default=False,
                        help='To disable dropout layers in the model.')
    parser.add_argument('--verbos', action='store_true', default=False,
                        help='Enable print.')
    parser.add_argument('--merge-plot', action='store_true', default=False,
                        help='Merge train losses and test losses.')
    # Add model name
    parser.add_argument('--model-name', type=str, default='cnn', metavar='M',
                        help='Model name (default: cnn)')

    args = parser.parse_args()

    if not args.verbos:
        print("""
        Verbos is disabled. Not much information is printed until the end of training. 
        Enable it by adding --verbos flag. This benefit for precise time measurement."
        """)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dv = "cuda" if use_cuda else "cpu"
    print(f"Using {dv} device")

    # prepare base filename
    suffix_filename = get_suffix_filename(args, dv)
    exp_dir = get_export_dir(args)

    if args.merge_plot:
        plot_all_losses(exp_dir, suffix_filename)
        return

    if args.disable_dropout:
        suffix_filename += '_dropout_disabled'
    
    if args.reverse_data:
        suffix_filename += '_reverse_data'


    if args.read_only:
        losses = pd.read_csv(f'{exp_dir}/losses_{suffix_filename}.csv')
        plot_losses(exp_dir, suffix_filename, losses, args.disable_dropout, args.reverse_data)
        trained_time_consumed = losses['train_time'].tolist()
        test_time_consumed = losses['test_time'].tolist()
        print_time(trained_time_consumed, test_time_consumed)
        return

    torch.manual_seed(args.seed)

    device = torch.device(dv)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.model_name == 'cnn':
        model = Net(args.disable_dropout).to(device)
    elif args.model_name == 'resnet':
        model = SimpleResNet(args.disable_dropout).to(device)
    else:
        raise ValueError(f"Model name {args.model_name} not supported")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    trained_time_consumed = []
    test_time_consumed = []
    train_losses = []
    test_losses = []

    if args.reverse_data:
        train_loader, test_loader = test_loader, train_loader
    
    for epoch in range(1, args.epochs + 1):
        tl, duration = train(args, model, device, train_loader, optimizer, epoch)
        train_losses.append(tl)
        trained_time_consumed.append(duration)
        tl, duration = test(args, model, device, test_loader)
        test_losses.append(tl)
        test_time_consumed.append(duration)
        scheduler.step()

    pd.DataFrame({'train_loss': train_losses, 
                  'test_loss': test_losses,
                  'train_time': trained_time_consumed,
                  'test_time': test_time_consumed}).to_csv(f'{exp_dir}/losses_{suffix_filename}.csv', index=False)

    print_time(trained_time_consumed, test_time_consumed)

    if args.save_model:
        torch.save(model.state_dict(), f"{exp_dir}/mnist_{args.model_name}_{suffix_filename}.pt")

if __name__ == '__main__':
    main()
