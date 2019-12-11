import argparse
import time
import math
import torch
import torch.nn as nn
import corpus
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./input', # /input
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--save', type=str,  default='./output/model_test.pt')
parser.add_argument('--opt', type=str,  default='SGD',
                    help='SGD, Adam, RMSprop, Momentum')
args = parser.parse_args()

torch.manual_seed(1111)

# Load data
corpus = corpus.Corpus(args.data)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size) # size(total_len//bsz, bsz)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# Build the model
interval = 200 # interval to report
ntokens = len(corpus.dictionary) # 10000
model = model.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout)

# Load checkpoint
if args.checkpoint != '':
    model = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

print(model)
criterion = nn.CrossEntropyLoss()

# Training code

def repackage_hidden(h):
    # detach
    return tuple(v.clone().detach() for v in h)


def get_batch(source, i):
    # source: size(total_len//bsz, bsz)
    seq_len = min(args.bptt, len(source) - 1 - i)
    #data = torch.tensor(source[i:i+seq_len]) # size(bptt, bsz)
    data = source[i:i+seq_len].clone().detach()
    target = source[i+1:i+1+seq_len].clone().detach().view(-1)
    #target = torch.tensor(source[i+1:i+1+seq_len].view(-1)) # size(bptt * bsz)
    return data, target


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    with torch.no_grad():
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size) #hidden size(nlayers, bsz, hdsize)
        for i in range(0, data_source.size(0) - 1, args.bptt):# iterate over every timestep
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            # model input and output
            # inputdata size(bptt, bsz), and size(bptt, bsz, embsize) after embedding
            # output size(bptt*bsz, ntoken)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss / len(data_source)


def train():
    # choose a optimizer

    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    # train_data size(batchcnt, bsz)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        opt.zero_grad()
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()

        total_loss += loss.data

        if batch % interval == 0 and batch > 0:
            cur_loss = total_loss / interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
opt = torch.optim.SGD(model.parameters(), lr=lr)
if args.opt == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    lr = 0.001
if args.opt == 'Momentum':
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
if args.opt == 'RMSprop':
    opt = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
    lr = 0.001

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            if args.opt == 'SGD' or args.opt == 'Momentum':
                lr /= 4.0
                for group in opt.param_groups:
                    group['lr'] = lr

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
