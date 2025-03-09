from __future__ import print_function
import argparse
import torch
from main import get_export_dir, get_suffix_filename

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

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    dv = "cuda" if use_cuda else "cpu"
    print(f"Using {dv} device")

    # prepare base filename
    suffix_filename = get_suffix_filename(args, dv)
    exp_dir = get_export_dir(args)
    
    plot_all_losses(exp_dir, suffix_filename)

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

if __name__ == '__main__':
    main()
