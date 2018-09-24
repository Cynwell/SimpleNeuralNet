import torch
from dataset import test_loader
import model_0


def test_model():
    # Hyper-parameters
    num_of_samples = 500
    batch_size = 50
    input_size = 1
    num_classes = 2

    # Initialize the model and its parameters.
    model = model_0.Model(input_size, num_classes)
    with open('model_0_parameters', 'rb') as f:
        model.load_state_dict(torch.load(f))

    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader(num_of_samples, batch_size):
            data = data.reshape(-1, input_size)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on {} samples: {}'.format(num_of_samples, correct / total))


def human_input_test_data():
    # Hyper-parameters
    input_size = 1
    num_classes = 2

    # Initialize the model and its parameters.
    model = model_0.Model(input_size, num_classes)
    with open('model_0_parameters', 'rb') as f:
        model.load_state_dict(torch.load(f))

    with torch.no_grad():
        while True:
            data = int(input('Input Data (0, 1): '))
            data = torch.Tensor([data])
            data = data.reshape(-1, input_size)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            print('Predicted label:', int(predicted[0].data))


def main():
    test_model()
    # human_input_test_data()


if __name__ == '__main__':
    main()
