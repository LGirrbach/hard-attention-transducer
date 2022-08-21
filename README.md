# End-to-End Training for Hard Attention Transducers

## Overview
This repository contains PyTorch-based implementations of end-to-end trainable transducers using hard attention
instead of soft attention.
There is one autoregressive and one non-autoregressive transducer.

### What are transducers?
Transducers are a type of sequence transduction model often used for string rewriting or morphology related tasks.
Instead of directly predicting the target sequence from the source sequence, as is the case in typical machine
translation and other sequence-to-sequence models, transducers predict edit operations.
The edit operations considered here are:
 * Delete: Remove the respective source symbol
 * Copy: Copy the respective source symbol to the target sequence
 * Substitution: Replace the respective source symbol by a target symbol
   (there is 1 substitution action for each symbol in the target alphabet)
 * Insertion: Predict a symbol in the target sequence
   (there is 1 insertion action for each symbol in the target alphabet)

For each symbol in the source sequence, the transducer predicts a number of edit operations, which then determine how
the source sequence is transformed to yield the predicted target sequence.

### What is hard attention?
Typically, sequence-to-sequence models use soft attention, where the result of attention is a weighted sum of all
attention keys (as in query-key-value-attention). In contrast, hard attention selects one and only one key to attend to.
In the context of transducers, this means that decoding of transduction actions always attends to exactly one source
symbol, instead of soft attention over source symbols.

### What is end-to-end training?
End-to-end training means the property of a neural model that all computations that are required to calculate the loss
are differentiable wrt. model parameters. An example of a non-end-to-end trainable model is the approach described
by [Makarov and Clematide (2018)](https://aclanthology.org/D18-1314/): Since their decoding strategy takes each
previously predicted edit action into account, this information must be available at training time, either by sampling
or by using an external aligner, which are both not differentiable computations (if we view the aligner as part of the
optimisation goal).

Note that hard attention is inherently a non-differentiable operation. However, end-to-end training of _monotonous_
hard attention (i.e. attention can only shift to the next following source symbol or stay in the current position) can
be done efficiently by marginalising over all possible alignments using dynamic programming. The main idea has been
described by [Yu et al. (2016)](https://aclanthology.org/D16-1138) and
[Libovický and Fraser (2022)](https://aclanthology.org/2022.spnlp-1.6).

## Model Descriptions
### Encoder
The implemented encoder is a BiLSTM encoder. Also, SOS and a EOS tokens are added to source sequences and the initial
hidden states are made trainable.
Important parameters of the encoder are (names as parameters in `make_settings` from [settings.py](settings.py):
 * `embedding_size`
 * `hidden_size`
 * `hidden_layers`
 * `dropout`

### Feature Encoder
In case you use feature, as is usually the case for e.g. morphological inflection tasks, set the `use_feature`
parameter in `make_settings` (from [settings.py](settings.py)) to `True`.
Features are a sequence of feature symbols, for example inflection tags.
The feature encoder also is a BiLSTM encoder
with trainable initial hidden states, but you can skip the LSTM by setting `features_num_layers` to `0`.
Then, each feature symbol is only embedded but not contextualised.

For each predicted edit action (autoregressive model) or source symbol (non-autoregressive model), feature symbol
encodings are combined to a single vector representing the entire feature symbol sequence. This implementation includes
2 methods to do so:
 * Mean, Max, Sum pooling: Ignores the encoder/decoder information and simply pools the encoded feature sequence
 * MLP, Dot Product Attention: Lets the encoder/decoder queries (soft) attend to feature encodings and encodes the 
   feature sequence by the resulting weighted sum of feature symbol encodings

Parameters of the feature encoder are:
  * `features_num_layers`: Number of LSTM layers in feature encoder, no LSTM if `0`
  * `features_pooling`: Type of feature sequence pooling, can be `'mean', ''sum', 'max', 'mlp', 'dot'`

Hidden size, embedding size, and dropout are the same as for the source sequence encoder.

### Autoregressive Decoder
The autoregressive decoder is a LSTM predicting the next edit action from the previous decoder hidden state,
last predicted target  symbol and optionally features. In contrast to
[Makarov and Clematide (2018)](https://aclanthology.org/D18-1314/), this implementation does not use the last
predicted action but the last predicted symbol, which avoids having to  provide ground-truth actions at training time.

During decoding, the decoder hidden state is only updates if a new symbol is predicted (i.e. no Delete action). Note
that the edit actions allow for online decoding of the predicted target sequence. Hard Attention starts with the first
(the SOS) symbol and shifts to the next symbol when predicting a Delete, Substitution, or CopyShift
(which is a shortcut for Copy followed by Delete) action.

At training time, the ground-truth target sequence is known, and so we can use teacher forcing to train the model.
Furthermore, we marginalise over possible alignments of source symbols to target symbols and possible edit operations.
Using dynamic programming, we calculate the probability of predicting target sequence prefix
$t_{1:n}$ from source sequence prefix $s_{1:m}$ recursively by
$$
P(t_{1:n}|s_{1:m}) =
P_{\text{del}}(s_m) \cdot P(t_{1:n}|s_{1:m-1}) +
\delta_{t_n == s_m} \cdot P_{\text{copy-shift}}(s_m) \cdot P(t_{1:n-1}|s_{1:m-1}) + 
P_{\text{sub}}(t_n|s_m) \cdot P(t_{1:n-1}|s_{1:m-1}) +
\delta_{t_n == s_m} \cdot P_{\text{copy}}(s_m) \cdot P(t_{1:n-1}|s_{1:m}) +
P_{\text{ins}}(t_n|s_m) \cdot P(t_{1:n-1}|s_{1:m})
$$
where $P_{\text{del}}, P_{\text{copy-shift}}, P_{\text{sub}}, P_{\text{copy}}, P_{\text{ins}}$ are the probabilities
for Delete, CopyShift, Substitution, Copy, and Insertion. $\delta_{t_n == s_m}$ is the indicator function stating
whether copying is possible (target symbol equals source symbol).

### Non-Autoregressive Decoder

## Usage
You need 3 ingredients to use this code:
First, make datasets
```python
from dataset import RawDataset

train_data = RawDataset(
    sources: List[List[str]]=train_sources,
    targets: List[List[str]]=train_targets,
    features: Optional[List[List[str]]] = train_features
)
development_data = RawDataset(
    sources: List[List[str]]=development_sources,
    targets: List[List[str]]=development_targets,
    features: Optional[List[List[str]]] = development_features
)
```
Here, `sources`, `targets` and `features` are datasets containing sequences of symbols encoded as strings.

Next, define settings:
```python
from settings import make_settings

settings = make_settings(
    use_features: bool = True,
    autoregressive: bool = True,
    name: str = 'test', 
    save_path: str = "./saved_models"
)
```
There are many hyperparameters, which are described in `settings.py`. The required arguments are `use_features`,
which tells the transducer whether to use provided features, `autoregressive`, which tells the transducer whether
to use the autoregressive or non-autoregressive model, and `name` and `save_path`, which are used to name and save
checkpoints. It is also recommended to pass your `device`.

Finally, you can train a model:
```python
from transducer import Transducer

model = Transducer(settings=settings)
model = model.fit(
    train_data: RawDataset=train_data,
    development_data: RawDataset=development_data
)

predictions = model.predict(test_sources: List[List[str]])
```
Predictions come as a list of a `namedtuple` called `TransducerPrediction`, which has 2 attributes,
namely the predicted symbols `prediction` and also the alignment `alignment` of predicted symbols and actions to
source symbols.

## References
 * Monotonic Hard Attention Transducers: [Makarov and Clematide (2018a)](https://aclanthology.org/C18-1008),
   [Makariv and Clematide (2018b)](https://aclanthology.org/D18-1314),
   [Clematide and Makarov (2021)](https://aclanthology.org/2021.sigmorphon-1.17),
   [Wehrli et al. (2022)](https://aclanthology.org/2022.sigmorphon-1.21)
 * End-to-End Training for Hard Attention: [Yu et al. (2016)](https://aclanthology.org/D16-1138),
   [Libovický and Fraser (2022)](https://aclanthology.org/2022.spnlp-1.6)
 * Non-Autoregressive Models: [Libovický and Helcl (2018)](https://aclanthology.org/D18-1336)
