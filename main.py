import argparse

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from lab9.model import Net
from lab9.nn import train, test

def main(args: argparse.Namespace):
    # Training settings

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = getattr(datasets, args.dataset)('../data', train=True, download=True,
                                               transform=transform)
    dataset2 = getattr(datasets, args.dataset)('../data', train=False,
                                               transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    epochs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        if args.plot:
            train_loss, test_loss, train_acc, test_acc = plot_helper(model, device, train_loader, test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            epochs.append(epoch)

        scheduler.step()

    if args.plot:
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.legend()
        plt.savefig(f'Loss_lr={args.lr}_batch={args.batch_size}_optimizer={args.optimizer}.png')
        plt.clf()
        plt.plot(epochs, train_accs, label="Training Accuracy")
        plt.plot(epochs, test_accs, label="Test Accuracy")
        plt.legend()
        plt.savefig(f'Accuracy_lr={args.lr}_batch={args.batch_size}_optimizer={args.optimizer}.png')
    if args.save_model:
        torch.save(model.state_dict(), args.dataset + ".pt")

if __name__ == '__main__':
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
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='For plotting the training/testing curves')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='For choosing which optimizer')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='For choosing a default dataset from torchvision')
    args = parser.parse_args()
    main(args)
