Project Title: Word-level language modeling FNN

Description: This example trains a multi-layer FNN on a language modeling task. By default, the training script uses the Wikitext-2 dataset, provided. The trained model can then be used by the generate script to generate new text.
The FNN model is trained using batch gradient descent with multiplicativeLR scheduler, which divides the learning rate by a factor of 4 when there is no improvement in perplexity. 

Example code:
python main.py --cuda --epochs 6           # Train a FNN on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied FNN on Wikitext-2 with CUDA
python generate.py

The main.py script accepts the following arguments:
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU,
                        Transformer)
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --bptt BPTT           sequence length
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --tied                tie the word embedding and softmax weights
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --onnx-export ONNX_EXPORT
                        path to export the final model in onnx format
  --nhead NHEAD         the number of heads in the encoder/decoder of the
                        transformer model
  --optimizer OPTIMIZER selection of optimizer, how the model trains
  --scheduler SCHEDULER selection of scheduler, how the learning rate changes between epochs

Aside from what was given, there is an addition of 2 new arguments, optimizer and scheduler.
Optimizers and schedulers are taken from the pytorch library. 
Optimizers include - SGD, ASGD, Adadelta, Adagrad, Adamax, Rprop, Adam, AdamW.
Schedulers include - LambdaLR, MultiplicativeLR, StepLR, MultistepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts.
Note: Pytorch schedulers will only work on pytorch optimizers. Running a --scheduler without a --optimizer, it will default back to batch gradient descent with multiplicativeLR scheduler.

Example code:
python main.py --cuda --epochs 6 --optimizer Adadelta
python main.py --cuda --epochs 6 --optimizer SGD --scheduler ExponentialLR
python main.py --cuda --epochs 6 --optimizer ASGD --scheduler CosineAnnealingLR --tied

Required Libraries:
-pytorch

How to use run code with google colab:
1. Place the files main.py, model.py, data.py, generate.py and 'data' folder into google drive
2. Change runtime type in google collab to GPU
3. Run the first code box, and link your google drive to google colab
4. Change the directory in the second code box to the location where the files are stored in google drive
5. Run the rest of the code

Contributors:
Ng Joshua
Mark Chua
Eric Ce

