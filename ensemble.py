import numpy as np

from typing import List
from typing import Dict
from typing import Optional
from transducer import Sequences
from transducer import Transducer
from inference import AlignmentPosition
from inference import TransducerPrediction


def _sequence_ensemble(predictions: List[TransducerPrediction], scores: Dict[int, float]) -> TransducerPrediction:
    votes = {
        i: sum([other_prediction.prediction == current_prediction.prediction for other_prediction in predictions])
        for i, current_prediction in enumerate(predictions)
    }
    max_votes = max(votes.values())
    winning_predictions = [index for index, index_votes in votes.items() if index_votes == max_votes]
    winning_prediction_index = winning_predictions[np.argmin([scores[idx] for idx in winning_predictions])]
    return predictions[winning_prediction_index]


def _position_ensemble(positions: List[List[AlignmentPosition]], scores: Dict[int, float]) -> List[AlignmentPosition]:
    prediction = []

    for position_predictions in positions:
        votes = {
            i: sum(
                [
                    other_prediction.predictions == current_prediction.predictions
                    for other_prediction in position_predictions
                ]
            )
            for i, current_prediction in enumerate(position_predictions)
        }
        max_votes = max(votes.values())
        winning_predictions = [index for index, index_votes in votes.items() if index_votes == max_votes]
        winning_prediction_index = winning_predictions[np.argmin([scores[idx] for idx in winning_predictions])]
        prediction.append(position_predictions[winning_prediction_index])

    return prediction


def ensemble_predict(transducers: List[Transducer], sources: Sequences, features: Optional[Sequences] = None,
                     tie_breaking_metric: str = "loss"):
    # Check parameters
    assert all([tie_breaking_metric in transducer.model.metrics for transducer in transducers])

    ensemble_predictions: List[List[TransducerPrediction]] = [
        transducer.predict(sources=sources, features=features) for transducer in transducers
    ]
    scores = {i: transducer.model.metrics[tie_breaking_metric] for i, transducer in enumerate(transducers)}

    ensembled_predictions_position = []
    ensembled_predictions_sequence = []

    for predictions in zip(*ensemble_predictions):
        # Sequence ensemble
        predictions = list(predictions)
        ensembled_predictions_sequence.append(_sequence_ensemble(predictions, scores=scores))

        # Position ensemble
        position_predictions = [[] for _ in predictions[0].alignment]

        for prediction in predictions:
            for position, position_prediction in enumerate(prediction.alignment):
                position_predictions[position].append(position_prediction)

        alignment = _position_ensemble(positions=position_predictions, scores=scores)
        prediction = []
        for position in alignment:
            prediction.extend(position.predictions)

        ensembled_predictions_sequence.append(TransducerPrediction(prediction=prediction, alignment=alignment))

    return ensembled_predictions_sequence, ensembled_predictions_position
