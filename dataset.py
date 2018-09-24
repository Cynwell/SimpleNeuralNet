import random
import torch


# Load data into the loop.
def train_loader(n, batch_size):
    x_list = torch.Tensor([])
    y_list = torch.LongTensor([])
    for i in range(n):
        a = random.randint(0, 1)
        x = torch.Tensor([a])
        y = torch.LongTensor([a])
        x_list = torch.cat((x_list, x), 0)
        y_list = torch.cat((y_list, y), 0)
        if (i + 1) % batch_size == 0:
            # Clear up the x_list and y_list so that previously passed data will not be accumulated.
            x_yield = x_list
            y_yield = y_list
            x_list = torch.Tensor([])
            y_list = torch.LongTensor([])
            yield (x_yield, y_yield)


# Load data into the loop.
def test_loader(n, batch_size):
    x_list = torch.Tensor([])
    y_list = torch.LongTensor([])
    for i in range(n):
        a = random.randint(0, 1)
        x = torch.Tensor([a])
        y = torch.LongTensor([a])
        x_list = torch.cat((x_list, x), 0)
        y_list = torch.cat((y_list, y), 0)
        if (i + 1) % batch_size == 0:
            yield (x_list, y_list)


def main():
    train_loader()


if __name__ == '__main__':
    main()
