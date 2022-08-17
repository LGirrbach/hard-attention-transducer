# End-to-end Training for Hard Attention Transducers

This repository contains PyTorch-based implementations of end-to-end trainable transducers using hard attention instead of soft attention.
There is one autoregressive and one non-autoregressive transducer.

## Usage
You need 3 ingredients to use this code:
First, make datasets
```
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
```
from settings import make_settings

settings = make_settings(
    use_features: bool = True,
    autoregressive: bool = True,
    name: str = 'test', 
    save_path: str = "./saved_models"
)
```
There are many hyperparameters, which are described in `settings.py`. The required arguments are `use_features`, which tells the transducer whether to use provided features, `autoregressive`, which tells the transducer whether to use the autoregressive or non-autoregressive model, and `name` and `save_path`, which are used to name and save checkpoints. It is also recommended to pass your `device`.

Finally, you can train a model:
```
from transducer import Transducer

model = Transducer(settings=settings)
model = model.fit(train_data: RawDataset=train_data, development_data: RawDataset=development_data)

predictions = model.predict(test_sources: List[List[str]])
```
Predictions come as a list of a `namedtuple` called `TransducerPrediction`, which has 2 attributes, namely the predicted symbols `prediction` and also the alignment `alignment` of predicted symbols and actions to source symbols. 
