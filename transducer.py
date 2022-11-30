from __future__ import annotations

from tqdm import tqdm
from typing import List
from trainer import train
from typing import Optional
from settings import Settings
from dataset import RawDataset
from trainer import load_model
from trainer import TrainedModel
from inference import TransducerPrediction
from inference import non_autoregressive_inference
from inference import soft_attention_greedy_sampling
from inference import autoregressive_greedy_sampling
from inference import soft_attention_beam_search_sampling
from inference import autoregressive_beam_search_sampling

Sequences = List[List[str]]


class Transducer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Optional[TrainedModel] = None

    @classmethod
    def load(cls, path: str) -> Transducer:
        model = load_model(path=path)
        transducer = cls(settings=model.settings)
        transducer.model = model

        return transducer

    def fit(self, train_data: RawDataset, development_data: Optional[RawDataset] = None) -> Transducer:
        self.model = train(train_data=train_data, development_data=development_data, settings=self.settings)
        return self

    def predict(self, sources: Sequences, features: Optional[Sequences] = None) -> List[TransducerPrediction]:
        if features is not None:
            assert len(sources) == len(features)

        if self.model is None:
            raise RuntimeError("Running inference with uninitialised model")

        # Make batches
        batches = []
        batch_size = self.settings.batch
        for batch_index in range(len(sources) // self.settings.batch + 1):
            start = batch_index * batch_size
            end = (batch_index + 1) * batch_size

            if features is not None:
                batches.append((sources[start:end], features[start:end]))
            else:
                batches.append((sources[start:end], None))

        batches = [batch for batch in batches if len(batch[0]) > 0]

        if self.settings.verbose:
            batches = tqdm(batches, desc="Prediction Progress")

        predictions = []

        for sources, features in batches:
            predictions.extend(self._predict_batch(sources=sources, features=features))

        return predictions

    def _predict_soft_attention_batch(self, sources: Sequences, features: Optional[Sequences] = None) \
            -> List[TransducerPrediction]:
        if self.settings.beam_search:
            return soft_attention_beam_search_sampling(
                model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
                num_beams=self.settings.num_beams, max_decoding_length=self.settings.max_decoding_length,
                sequences=sources, features=features
            )
        else:
            return soft_attention_greedy_sampling(
                model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
                max_decoding_length=self.settings.max_decoding_length, sequences=sources, features=features
            )

    def _predict_autoregressive_batch(self, sources: Sequences, features: Optional[Sequences] = None) \
            -> List[TransducerPrediction]:
        if self.settings.beam_search:
            return autoregressive_beam_search_sampling(
                model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
                num_beams=self.settings.num_beams, max_decoding_length=self.settings.max_decoding_length,
                sequences=sources, features=features
            )
        else:
            return autoregressive_greedy_sampling(
                model=self.model.model, source_vocabulary=self.model.source_vocabulary,
                target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
                max_decoding_length=self.settings.max_decoding_length, sequences=sources, features=features
            )

    def _predict_non_autoregressive_batch(self, sources: Sequences, features: Optional[Sequences] = None) \
            -> List[TransducerPrediction]:
        return non_autoregressive_inference(
            model=self.model.model, source_vocabulary=self.model.source_vocabulary,
            target_vocabulary=self.model.target_vocabulary, feature_vocabulary=self.model.feature_vocabulary,
            sequences=sources, features=features
        )

    def _predict_batch(self, sources: Sequences, features: Optional[Sequences] = None) -> List[TransducerPrediction]:
        if self.settings.model == "soft-attention":
            return self._predict_soft_attention_batch(sources=sources, features=features)
        elif self.settings.model == "autoregressive":
            return self._predict_autoregressive_batch(sources=sources, features=features)
        elif self.settings.model == "non-autoregressive":
            return self._predict_non_autoregressive_batch(sources=sources, features=features)
        else:
            raise ValueError(f"Unknown model: {self.settings.model}")
