## A RNN model to classify protein sequences - in Python & TensorFlow

 
A character-level RNN (Basic LSTM, LSTM & GRU) model to classsify protein sequences. It takes a csv file containig at least two columns: sequence (protein sequences) and label (what protein family or cluster the sequence belongs to) as input, splits the data into 80% training : 20% test, trains the model, classifies the protein sequences in the test dataset, and finally spits out the train & test accuracy measures. Users can tune the model hyperparameters, such as the type of the cell (Basic LSTM, LSTM & GRU), number of layers, number of neurons in each layer, dropout value, learning rate, etc., in the main function. A toy dataset is provided in the data folder.

## Requirements

- Python 3
- Tensorflow > 1.0.1

## Training

```
python rnn_classifier.py
```
## Help

```
python rnn_classifier.py -h
```
