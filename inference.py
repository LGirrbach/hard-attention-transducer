import torch
import torch.nn as nn

from typing import List
from copy import deepcopy
from typing import Optional
from collections import namedtuple
from models.base import TransducerModel
from torch.nn.utils.rnn import pad_sequence
from vocabulary import SourceVocabulary, TransducerVocabulary
from actions import Deletion, Copy, CopyShift, Insertion, Substitution, Noop


Beam = namedtuple(
    "Beam",
    ["source_index", "position", "hidden", "predictions", "alignments", "score"]
)
AlignmentPosition = namedtuple("AlignmentPosition", ["symbol", "actions", "predictions"])
TransducerPrediction = namedtuple("TransducerPrediction", ["prediction", "alignment"])


def non_autoregressive_inference(model: TransducerModel, source_vocabulary: SourceVocabulary,
                                 target_vocabulary: TransducerVocabulary, sequences: List[List[str]],
                                 feature_vocabulary: Optional[SourceVocabulary] = None,
                                 features: Optional[List[List[str]]] = None,
                                 max_decoding_length: int = 20) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.device)
    tau: int = model.tau
    if tau is None:
        tau = max_decoding_length

    # Index sources
    sequences = [
        [source_vocabulary.SOS_TOKEN] + source + [source_vocabulary.EOS_TOKEN] for source in sequences
    ]
    source_lengths = torch.tensor([len(source) for source in sequences]).long()
    sources = [torch.tensor(source_vocabulary.index_sequence(source)).long() for source in sequences]
    sources = pad_sequence(sources, padding_value=0, batch_first=True)

    # Index features
    if model.use_features:
        features = [
            [feature_vocabulary.SOS_TOKEN] + feats + [feature_vocabulary.EOS_TOKEN] for feats in features
        ]
        feature_lengths = torch.tensor([len(feats) for feats in features]).long()
        features = [torch.tensor(feature_vocabulary.index_sequence(feats)).long() for feats in features]
        features = pad_sequence(features, padding_value=0, batch_first=True)
    else:
        features = None
        feature_lengths = None

    # Get predictions
    with torch.no_grad():
        scores = model(
            sources=sources, lengths=source_lengths, features=features, feature_lengths=feature_lengths,
            tau=max_decoding_length
        )
        scores = scores.reshape(len(sequences), -1, len(target_vocabulary))
        predictions = scores.argmax(dim=-1).detach().cpu()

    # Decode predictions
    hypotheses = []

    for sequence, predicted_action_idx in zip(sequences, predictions):
        # Select only relevant predictions
        predicted_action_idx = predicted_action_idx[:tau * len(sequence)]
        predicted_action_idx = predicted_action_idx.reshape(-1, tau)
        predicted_action_idx = predicted_action_idx.tolist()

        action_history = []
        sequence_prediction = []

        for symbol, predicted_actions in zip(sequence, predicted_action_idx):
            current_actions = []
            current_predictions = []

            for action_idx in predicted_actions:
                action = target_vocabulary[action_idx]
                current_actions.append(action)

                if isinstance(action, Insertion):
                    current_predictions.append(action.token)

                elif isinstance(action, Substitution):
                    current_predictions.append(action.token)
                    break

                elif isinstance(action, Copy):
                    current_predictions.append(symbol)

                elif isinstance(action, CopyShift):
                    current_predictions.append(symbol)
                    break

                elif isinstance(action, Deletion):
                    break

            action_history.append({"symbol": symbol, "actions": current_actions, "predictions": current_predictions})
            sequence_prediction.extend(current_predictions)

        hypotheses.append((sequence_prediction, action_history))

    # Reformat hypotheses
    predictions = []
    for prediction, alignment in hypotheses:
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], actions=position["actions"], predictions=position["predictions"]
            ) for position in alignment
        ]
        predictions.append(TransducerPrediction(prediction=prediction, alignment=alignment))

    return predictions


def autoregressive_greedy_sampling(model: nn.Module, source_vocabulary: SourceVocabulary,
                                   target_vocabulary: TransducerVocabulary, sequences: List[List[str]],
                                   feature_vocabulary: Optional[SourceVocabulary] = None,
                                   features: Optional[List[List[str]]] = None,
                                   max_decoding_length: int = 70) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.device)

    # Index sources
    sequences = [
        [source_vocabulary.SOS_TOKEN] + source + [source_vocabulary.EOS_TOKEN] for source in sequences
    ]
    source_lengths = torch.tensor([len(source) for source in sequences]).long()
    sources = [torch.tensor(source_vocabulary.index_sequence(source)).long() for source in sequences]
    sources = pad_sequence(sources, padding_value=0, batch_first=True)

    # Index features
    if model.use_features:
        features = [
            [feature_vocabulary.SOS_TOKEN] + feats + [feature_vocabulary.EOS_TOKEN] for feats in features
        ]
        feature_lengths = torch.tensor([len(feats) for feats in features]).long()
        features = [torch.tensor(feature_vocabulary.index_sequence(feats)).long() for feats in features]
        features = pad_sequence(features, padding_value=0, batch_first=True)
    else:
        features = None
        feature_lengths = None

    # Run encoder
    with torch.no_grad():
        source_encodings = model.encode(sources, source_lengths)

    hypotheses = [[target_vocabulary.SOS_TOKEN] for _ in sequences]  # Store generated predictions
    action_histories = [
        [{"symbol": symbol, "actions": [], "predictions": []} for symbol in sequence] for sequence in sequences
    ]
    sampled_tokens = [[target_vocabulary.get_symbol_index(target_vocabulary.SOS_TOKEN)] for _ in sequences]
    sampled_tokens = torch.tensor(sampled_tokens).long()
    positions = [0 for _ in sequences]

    hidden = None
    step_num = 0

    while (
            any(position < length.item() for position, length in zip(positions, source_lengths)) and
            step_num < max_decoding_length
    ):
        step_num += 1
        positions = [min(position, source_encodings.shape[1] - 1) for position in positions]

        # Get next predicted tokens
        with torch.no_grad():
            target_encodings, (old_hidden, new_hidden) = model.decode(
                sampled_tokens, torch.ones(len(sequences)), source_encodings, source_lengths, hidden=hidden
            )
            current_source_positions = source_encodings[torch.arange(0, len(sequences)), positions, :]

            classifier_inputs = torch.cat(
                [current_source_positions, target_encodings.squeeze(1)], dim=-1
            )

            if model.use_features:
                feature_encodings = model.feature_encoder(
                    features, feature_lengths, classifier_inputs.unsqueeze(1)
                )
                classifier_inputs = torch.cat([classifier_inputs, feature_encodings.squeeze(1)], dim=-1)

            scores = model.classifier(classifier_inputs)
            predictions = scores.argmax(dim=-1).detach().cpu().tolist()

        sampled_hidden = []

        for sentence_idx, prediction in enumerate(predictions):
            # If already done, ignore prediction
            if (
                    positions[sentence_idx] >= source_lengths[sentence_idx].item() or
                    (
                            len(hypotheses[sentence_idx]) > 0 and
                            hypotheses[sentence_idx][-1] == target_vocabulary.EOS_TOKEN
                    )
            ):
                sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                if hypotheses[sentence_idx][-1] != target_vocabulary.EOS_TOKEN:
                    hypotheses[sentence_idx].append(target_vocabulary.EOS_TOKEN)

            else:
                position = positions[sentence_idx]
                predicted_action = target_vocabulary[prediction]

                if isinstance(predicted_action, Copy):
                    sampled_token = sequences[sentence_idx][position]
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)

                elif isinstance(predicted_action, CopyShift):
                    sampled_token = sequences[sentence_idx][position]
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)
                    positions[sentence_idx] += 1

                elif isinstance(predicted_action, Deletion):
                    sampled_hidden.append((old_hidden[0][:, sentence_idx], old_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    positions[sentence_idx] += 1

                elif isinstance(predicted_action, Substitution):
                    sampled_token = predicted_action.token
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)
                    positions[sentence_idx] += 1

                elif isinstance(predicted_action, Insertion):
                    sampled_token = predicted_action.token
                    hypotheses[sentence_idx].append(sampled_token)
                    sampled_hidden.append((new_hidden[0][:, sentence_idx], new_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    action_histories[sentence_idx][position]["predictions"].append(sampled_token)

                elif isinstance(predicted_action, Noop):
                    sampled_hidden.append((old_hidden[0][:, sentence_idx], old_hidden[1][:, sentence_idx]))
                    action_histories[sentence_idx][position]["actions"].append(predicted_action)
                    positions[sentence_idx] += 1

                else:
                    raise RuntimeError(f"Sampled invalid action: {predicted_action}")

        h_0, c_0 = zip(*sampled_hidden)
        h_0, c_0 = torch.stack(h_0), torch.stack(c_0)
        h_0, c_0 = h_0.transpose(0, 1), c_0.transpose(0, 1)
        hidden = (h_0, c_0)
        sampled_tokens = [
            [target_vocabulary.get_symbol_index(hypothesis[-1])] for hypothesis in hypotheses
        ]
        sampled_tokens = torch.tensor(sampled_tokens).long()

    # Reformat predictions
    predictions = []
    for prediction, alignment in zip(hypotheses, action_histories):
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], actions=position["actions"], predictions=position["predictions"]
            ) for position in alignment
        ]
        predictions.append(TransducerPrediction(prediction=prediction[1:], alignment=alignment))

    return predictions


def autoregressive_beam_search_sampling(model: nn.Module, source_vocabulary: SourceVocabulary,
                                        target_vocabulary: TransducerVocabulary, sequences: List[List[str]],
                                        max_decoding_length: int = 70, num_beams: int = 5,
                                        feature_vocabulary: Optional[SourceVocabulary] = None,
                                        features: Optional[List[List[str]]] = None) -> List[TransducerPrediction]:
    model = model.eval()
    model = model.to(model.device)

    # Index sources
    sequences = [
        [source_vocabulary.SOS_TOKEN] + source + [source_vocabulary.EOS_TOKEN] for source in sequences
    ]
    source_lengths = torch.tensor([len(source) for source in sequences]).long()
    sources = [torch.tensor(source_vocabulary.index_sequence(source)).long() for source in sequences]
    sources = pad_sequence(sources, padding_value=0, batch_first=True)

    # Index features
    if model.use_features:
        features = [
            [feature_vocabulary.SOS_TOKEN] + feats + [feature_vocabulary.EOS_TOKEN] for feats in features
        ]
        feature_lengths = torch.tensor([len(feats) for feats in features]).long()
        features = [torch.tensor(feature_vocabulary.index_sequence(feats)).long() for feats in features]
        features = pad_sequence(features, padding_value=0, batch_first=True)
    else:
        features = None
        feature_lengths = None

    # Run encoder
    with torch.no_grad():
        source_encodings = model.encode(sources, source_lengths)

    # Initialise beams
    beams = dict()
    sos_token = target_vocabulary.SOS_TOKEN
    eos_token = target_vocabulary.EOS_TOKEN

    for source_index, source in enumerate(sequences):
        beams[source_index] = []

        beam = Beam(
            source_index=source_index, position=0, hidden=None, predictions=[sos_token],
            alignments=[{"symbol": symbol, "actions": [], "predictions": []} for symbol in source],
            score=0.0
        )
        beams[source_index].append(beam)

    # Initialise criterion to decide whether beam is finished
    def is_finished(bm: Beam) -> bool:
        has_eos = bm.predictions[-1] == eos_token
        empty_buffer = bm.position >= len(sequences[bm.source_index])

        return has_eos or empty_buffer

    # Helper function to retrieve all beams from grouped dictionary
    def get_all_beams():
        all_beams = []
        for grouped_beams in beams.values():
            all_beams.extend(grouped_beams)

        return all_beams

    step_num = 0

    class Hypotheses:
        def __init__(self):
            self.hypotheses = [[] for _ in sequences]

        def add(self, bm: Beam, s_index: int):
            insertion_index = 0

            for stored_beam_score, stored_beam in self.hypotheses[s_index]:
                if bm.score < stored_beam_score:
                    insertion_index += 1
                else:
                    break

            if insertion_index <= num_beams:
                self.hypotheses[s_index].insert(insertion_index, (bm.score, bm))

        def get_best_score(self, s_index: int) -> float:
            if len(self.hypotheses[s_index]) == 0:
                return -torch.inf
            return self.hypotheses[s_index][0][0]

    hypotheses = Hypotheses()

    while len(get_all_beams()) > 0 and step_num < max_decoding_length:
        # Increase step counter
        step_num += 1

        # Get next predicted tokens
        with torch.no_grad():
            # Collect all active beams
            current_beams = get_all_beams()

            # Collect previously sampled symbols from active beams
            sampled_symbols = [beam.predictions[-1] for beam in current_beams]
            sampled_symbols = [[target_vocabulary.get_symbol_index(symbol)] for symbol in sampled_symbols]
            sampled_symbols = torch.tensor(sampled_symbols).long()

            # Collect source encodings of active beams
            current_source_encodings = [source_encodings[beam.source_index] for beam in current_beams]
            current_source_encodings = torch.stack(current_source_encodings)

            # Collect source lengths of active beams
            current_source_lengths = [source_lengths[beam.source_index] for beam in current_beams]
            current_source_lengths = torch.stack(current_source_lengths).long().flatten()

            # Collect decoder hidden states of active beams
            if step_num > 1:
                current_hidden = [beam.hidden for beam in current_beams]
                current_h0, current_c0 = zip(*current_hidden)
                current_h0 = torch.stack(current_h0)
                current_c0 = torch.stack(current_c0)
                current_h0 = current_h0.transpose(0, 1)
                current_c0 = current_c0.transpose(0, 1)
                current_hidden = (current_h0, current_c0)
            else:
                current_hidden = None

            # Calculate new decoder hidden states
            target_encodings, (old_hidden, new_hidden) = model.decode(
                sampled_symbols, torch.ones(len(current_beams)), current_source_encodings, current_source_lengths,
                hidden=current_hidden
            )

            # Extract source positions for hard attention
            batch_indexer = torch.arange(0, len(current_beams))
            current_positions = [beam.position for beam in current_beams]
            current_contexts = current_source_encodings[batch_indexer, current_positions, :]

            # Calculate prediction scores for each beam
            classifier_inputs = torch.cat(
                [current_contexts, target_encodings.squeeze(1)], dim=-1
            )

            if model.use_features:
                current_features = features[[beam.source_index for beam in current_beams]]
                current_feature_lengths = feature_lengths[[beam.source_index for beam in current_beams]]

                feature_encodings = model.feature_encoder(
                    current_features, current_feature_lengths, classifier_inputs.unsqueeze(1)
                )
                classifier_inputs = torch.cat([classifier_inputs, feature_encodings.squeeze(1)], dim=-1)

            scores = model.classifier(classifier_inputs)
            scores = model.normalise_scores(scores)
            scores = scores.detach().cpu()

        # Update beams
        new_beams = {source_index: [] for source_index in beams.keys()}

        for idx, (beam_scores, beam) in enumerate(zip(scores, current_beams)):
            branch_counter = 0
            score_rank = 0
            beam_scores = beam_scores.flatten()
            sorted_score_indices = torch.argsort(beam_scores, descending=True)

            while branch_counter < num_beams and score_rank < len(sorted_score_indices):
                predicted_index = sorted_score_indices[score_rank].item()
                score = beam_scores[predicted_index].item()

                predicted_action = target_vocabulary[predicted_index]

                if (
                        isinstance(predicted_action, Copy) or
                        isinstance(predicted_action, CopyShift) or
                        isinstance(predicted_action, Substitution) or
                        isinstance(predicted_action, Insertion)
                ):
                    if isinstance(predicted_action, CopyShift):
                        sampled_symbol = sequences[beam.source_index][beam.position]
                        position_update = 1

                    elif isinstance(predicted_action, Copy):
                        sampled_symbol = sequences[beam.source_index][beam.position]
                        position_update = 0

                    elif isinstance(predicted_action, Substitution):
                        sampled_symbol = predicted_action.token
                        position_update = 1

                    else:
                        sampled_symbol = predicted_action.token
                        position_update = 0

                    # Make updated predictions
                    predictions = deepcopy(beam.predictions) + [sampled_symbol]

                    # Make updated alignment history
                    alignment = deepcopy(beam.alignments)
                    alignment[beam.position]["actions"].append(predicted_action)
                    alignment[beam.position]["predictions"].append(sampled_symbol)

                    # Get hidden
                    hidden = (new_hidden[0][:, idx], new_hidden[1][:, idx])

                    # Make updated beam
                    new_beam = Beam(
                        source_index=beam.source_index,
                        position=beam.position + position_update,
                        hidden=hidden,
                        predictions=predictions,
                        alignments=alignment,
                        score=beam.score + score
                    )

                elif predicted_action.is_deletion() or predicted_action.is_noop():
                    hidden = (old_hidden[0][:, idx], old_hidden[1][:, idx])

                    # Make updated alignment history
                    alignment = deepcopy(beam.alignments)
                    alignment[beam.position]["actions"].append(predicted_action)

                    new_beam = Beam(
                        source_index=beam.source_index,
                        position=beam.position + 1,
                        hidden=hidden,
                        predictions=deepcopy(beam.predictions),
                        alignments=alignment,
                        score=beam.score + score
                    )

                else:
                    raise RuntimeError(f"Illegal action sampled: {predicted_action}")

                if is_finished(new_beam) or step_num >= max_decoding_length:
                    hypotheses.add(bm=new_beam, s_index=new_beam.source_index)
                elif new_beam.score >= hypotheses.get_best_score(s_index=new_beam.source_index):
                    new_beams[beam.source_index].append(new_beam)
                    branch_counter += 1
                else:
                    branch_counter += 1

                score_rank += 1

        beams = {
            source_index: list(sorted(beam_candidates, key=lambda bm: -bm.score))[:num_beams]
            for source_index, beam_candidates in new_beams.items()
        }

    predictions = []
    for source_predictions in hypotheses.hypotheses:
        _, best_hypothesis = max(source_predictions, key=lambda hypothesis: hypothesis[0])

        prediction = best_hypothesis.predictions
        alignment = [
            AlignmentPosition(
                symbol=position["symbol"], predictions=position["predictions"], actions=position["actions"]
            )
            for position in best_hypothesis.alignments
        ]
        predictions.append(TransducerPrediction(prediction=prediction[1:], alignment=alignment))

    return predictions
