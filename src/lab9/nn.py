import torch
import torch.nn.functional as F


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({round(100 * batch_idx / len(train_loader))}%)]\tLoss: {round(loss.item(), 4)}')
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f'\nTest set: Average loss: {round(test_loss, 4)}, Accuracy: {correct}/{len(test_loader.dataset)} ({round(100. * correct / len(test_loader.dataset))}%)\n')


def plot_helper(model, device, train_loader, test_loader):
    """
    Plot helper that recomputes loss and acccuracyc and returns it

    Parameters
    ----------
    model
    device
    train_loader
    test_loader

    Returns
    -------

    """
    model.eval()
    test_loss = 0
    train_loss = 0
    train_correct = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_correct += pred.eq(target.view_as(pred)).sum().item()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            train_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    train_loss /= len(train_loader.dataset)
    test_acc = 100. * test_correct / len(test_loader.dataset)
    train_acc = 100. * train_correct / len(train_loader.dataset)

    return train_loss, test_loss, train_acc, test_acc
