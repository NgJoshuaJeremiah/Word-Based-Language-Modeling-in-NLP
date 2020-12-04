# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model') #initialize parameters for access using argparse
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='FNN',
                    help='type of recurrent net (FNN, RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200, #original 200
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,  #default=20
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', #20 for LSTM
                    help='batch size')
parser.add_argument('--bptt', type=int, default=7, #35 for LSTM
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--optimizer', type=str, default='none',
                    help='type of optimizer (SGD, Adagrad, Adadelta, Adam, RMSprop, etc)')
parser.add_argument('--scheduler', type=str, default='none',
                    help = 'type of scheduler (LambdaLR, CosineAnnealing, CyclicLR, etc)')

args = parser.parse_args() #create parser object

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed) #set random seed for pytorch RNG
if torch.cuda.is_available(): #check for CUDA
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu") #set device to run on

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data) #takes data, tokenizes them and adds the words into a corpus(dictionary)
# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size) #arranges train data into columns of args.batch_size
val_data = batchify(corpus.valid, eval_batch_size) #arranges valid data into columns of eval_batch_size
test_data = batchify(corpus.test, eval_batch_size) #arranges test data into columns of eval_batch_size

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary) #extract number of tokens in the corpus into a variable
#selection of model
if args.model == 'Transformer':
    model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device) #initialize model to transformer
elif args.model == 'LSTM':
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device) #initializa model to RNN
else:
    model = model.FNNModel(ntokens, args.emsize, args.nhid, args.tied).to(device) #initialize model to FNN by default

criterion = nn.NLLLoss() #use negative log likelihood for loss function

#selection of optimizer
if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("Running SGD optimizer")
elif args.optimizer == 'ASGD':
    optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    print("Running ASGD optimizer")
elif args.optimizer == 'Adadelta':
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.5, eps=1e-6)
    print("Running Adadelta optimizer")
elif args.optimizer == 'Adagrad':
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    print("Running Adagrad optimizer")
elif args.optimizer == 'Adamax':
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    print("Running Adamax optimizer")
elif args.optimizer == 'Rprop':
    optimizer = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    print("Running Rprop optimizer")
elif args.optimizer == 'RMSprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    print("Running RMSprop optimizer")
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    print("Running Adam optimizer")
elif args.optimizer == 'AdamW':
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
    print("Running AdamW optimizer")
else:
    optimizer = 'none'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#selection of scheduler
if args.scheduler!='none':
    if args.scheduler == 'LambdaLR':
        lmbda = lambda epoch: 0.95 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda, last_epoch=-1)
        print("Running LambdaLR scheduler")
    elif args.scheduler == 'MultiplicativeLR':
        lmbda = lambda epoch: 0.95
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda, last_epoch=-1, verbose=False)
        print("Running MultiplicativeLR scheduler")
    elif args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        print("Running StepLR scheduler")
    elif args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,5], gamma=0.5, last_epoch=-1)
        print("Running MultiStepLR scheduler")
    elif args.scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1)
        print("Running ExponentialLR scheduler")
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=0, last_epoch=-1)
        print("Running CosineAnnealingLR scheduler")
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        print("Running ReduceLROnPlateau scheduler")
    elif args.scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, get_lr(optimizer), get_lr(optimizer)*5, step_size_up=2000, step_size_down=0, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        print("Running CyclicLR scheduler")
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1, eta_min=0, last_epoch=-1)
        print("Running CosineAnnealingWarmRestarts scheduler")
    else:
        scheduler='none'
else:
    scheduler='none'
###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# def get_batch(source, i): #retrieves batch i from source data, batch size is based on length args.bptt
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target

def get_batch(source, i): #retrieves batch i from source data, number of data points retrieved is based on args.bptt
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary) #get length of dictionary
    if args.model == 'LSTM':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            elif args.model == 'LSTM':
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            else:
                data.type(torch.FloatTensor)
                output = model(data)
                output = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output, targets).item() #calculating total loss
    return total_loss / (len(data_source) - 1) #returned average loss


def train():
    # Turn on training mode which enables dropout.
    global lr
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary) #get length of dictionary
    if args.model == 'LSTM':
        hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad() #reset gradient to prevent accumulation of gradient
        if args.optimizer!='none':
            optimizer.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        elif args.model == 'LSTM':
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        else:
            output = model(data)
            output = output.view(-1, ntokens)

        loss = criterion(output, targets) #calculate loss
        loss.backward() #backward propagation
        if args.optimizer!='none':
            optimizer.step()
            lr=get_lr(optimizer)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        #printing of details
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        if args.scheduler != 'none':
            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
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
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)