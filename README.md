# SimpleNeuralNet
A basic and extremely simple neural net example for beginners.

### Description:

This program generates 2 classes of data as follows:

| Label | Value |
| ----- | ------|
|  0    |   0   |
|  1    |   1   |

When it is on training, both values and labels are fed to the model for adjusting weights.
When it is on testing, only values are given and the model would predict its corresponding labels.

Although this program seems meaningless, it has a extrememly simple structure so that everything are quite intuitive and easy to learn.


### Download link:

##### SimpleNeuralNet.zip v1.0
https://github.com/Cynwell/SimpleNeuralNet/files/2412189/SimpleNeuralNet.zip


### File description:

**main.py**: The main program starts from here. It would train the neural net and test the neural nets accuracy.

**model_0.py**: Defines the neural net structure.

**train_model.py**: Details about how to train a neural net.

**test_model.py**: Details about testing accuracy of the neural net.

**dataset.py**: Loading data to train_model and test_model. Datas are generated within this file.

After downloaded and trained files, a file '**model_0_parameters**' would be generated to store weightings of the neural net.

Enjoy :)
