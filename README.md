# language_modeling_lstm
PTB language modeling (RNN/LSTM, Pytorch)

A reproduction of Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329). Use `main.py` to train a RNN to predict words based on a given sequence of words and apply dropout to LSTM model to reduce overfitting. The `main.py` accepts the following optional arguments.

```
--data            location of the training data
--checkpoint      loading the existing model
--emsize          embedding size
--nhid            the dimension of hidden layers
--nlayers         the number of layers
--lr              learning rate
--clip            gradient clipping
--epochs          epochs number
--batch_size      batch size
--bptt            sequence length
--dropout         dropout
--save            location to save the current model
--opt             choose a optimizer (SGD, Momentum, Adam, RMSprop)
```

