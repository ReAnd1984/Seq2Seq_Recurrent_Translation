# Seq2Seq Recurrent Translation

![Logo](https://camo.githubusercontent.com/3225433dd177ce8d8255c760ab7aefbdba83107545b62d17f4459cde05ecedd5/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f74662d6167656e7473)

In this exercise, pairs of sentences will be used, one in English and one in Italian, taken from the "Tatoeba Project" (https://tatoeba.org/it/) where various users continuously contribute to providing examples of translations. With this dataset, it will be possible to train a Seq2Seq model based on a recurrent Encoder-Decoder to implement an automatic translator.

## Data Preprocessing

- The dataset is loaded into a Pandas DataFrame.
- Preprocessing functions are applied to the Input and Target columns to tokenize the sentences.

## Tokenization

- Sequences of words are converted into sequences of integers using the Keras Tokenizer object.
- Two tokenizers are used: one for input sequences and one for target sequences.

## Vocabulary Creation

- Vocabularies are derived from the tokenizers to map words to indices and vice versa.

## Padding

- Null (zero) values are added to the end of each sentence to make all sentences the same length, enabling batch training.

## Dataset Management

- The tf.data library is used to manage the dataset for training. Batches of examples are created and utilized during training.

## Model Architecture

The Seq2Seq model consists of three main components:

- **Encoder**: Based on a recurrent layer, it receives a word from the input sequence at each time step, producing an output and a state for each time step. The state at each time step t is used at time step t+1 along with the input related to time step t+1.
- **Encoder vector**: It is the state produced as output by the recurrent layers after processing the entire input sequence. During training, the network strives to produce the best possible information so that the subsequent decoder block can perform optimally.
- **Decoder**: Similar to the encoder block, it takes the encoder vector as input and produces output representing the probability corresponding to each token in the vocabulary.

## Cost Function and Metrics

- A custom cost function is defined to avoid considering padding values (0s) when calculating the error. A similar approach is applied for calculating the model accuracy.

## Training

- The encoder is called for each batch of input sequences, producing the encoded vector.
- The initial state of the decoder is set equal to the encoded vector.
- The decoder is called, passing the target sequence as input, starting from the second element to eliminate the `<sos>` token.
- Loss and accuracy are calculated for each batch of examples, and model parameters are updated.

## Teacher Forcing

Teacher Forcing is utilized during training, where the predicted output at time step t-1 is used as input for each subsequent time step t. This technique accelerates training in recurrent architectures.

## Evaluation

The output log and training results are evaluated to visualize information regarding the metrics of interest.

## Testing

- The model is tested by loading the last checkpoint of the previously saved training.
- Predictions are made by repeatedly calling the encoder and decoder until the `<eos>` token or the maximum length limit is reached.
- Testing examples are taken from the training dataset.
