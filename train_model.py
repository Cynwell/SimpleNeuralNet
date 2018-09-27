import torch
import torch.nn as nn
import model_0
from dataset import train_loader


def train_model():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper-parameters
    num_of_samples = 50
    input_size = 1
    num_classes = 2
    num_epochs = 30
    batch_size = 10
    learning_rate = 0.1

    # Initialize the model with pre-trained parameters.
    model = model_0.Model(input_size, num_classes)

    # Load model parameters.
    try:
        with open('model_0_parameters', 'rb') as f:
            print('Retrieving data...')
            model_parameters = torch.load(f)
            model.load_state_dict(model_parameters)
            print('Data loaded.')
    except FileNotFoundError:
        print('Data file does not exist.')
    finally:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_steps = num_of_samples // batch_size
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader(num_of_samples, batch_size)):
            # print('data shape: {}, labels shape: {}'.format(data.shape, labels.shape))
            data = data.reshape(batch_size, -1)
            # print('Data: {}, Labels: {}'.format(data, labels))
            # Forward pass. Get the predicted output from the model.
            outputs = model(data)
            # Evaluate the loss.
            # print('outputs shape:', outputs.shape)
            # print('outputs:', outputs)
            # print('labels shape:', labels.shape)
            # print('labels:', labels)
            loss = criterion(outputs, labels)
            # Backward pass. Optimize the weightings.
            optimizer.zero_grad()  # Why zero_grad()? Clear what accumulated gradient of mini-batch?
            loss.backward()
            optimizer.step()
            # Display progress.
            if (i + 1) % 10 == 0:
                print('Epoch {}, Step[{}/{}], Loss:{}'.format(epoch, i + 1, total_steps, loss.item()))

    # Save model parameters.
    with open('model_0_parameters', 'wb') as f:
        print('Saving data...')
        torch.save(model.state_dict(), f)
        print('Data saved.')


def main():
    train_model()


if __name__ == '__main__':
    main()
