import os
import torch
import numpy as np
import editdistance
import torch.nn as nn

from typing import List
from typing import Tuple
from torch import Tensor
from logger import logger
from dataset import Batch
from typing import Optional
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import AdamW
from settings import Settings
from dataset import RawDataset
from collections import namedtuple
from loss import soft_attention_loss
from models import SoftAttentionModel
from models.base import TransducerModel
from torch.utils.data import DataLoader
from models import NonAutoregressiveLSTM
from dataset import TransducerDatasetTrain
from models import LSTMEncoderDecoderModel
from torch.nn.utils import clip_grad_value_
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ExponentialLR
from loss import autoregressive_transduction_loss
from inference import non_autoregressive_inference
from inference import soft_attention_greedy_sampling
from inference import autoregressive_greedy_sampling
from loss import non_autoregressive_transduction_loss
from dataset import AutoregressiveTransducerDatasetTrain
from dataset import NonAutoregressiveTransducerDatasetTrain
from vocabulary import SourceVocabulary, TransducerVocabulary

from loss import fast_autoregressive_transduction_loss


Sequence = List[str]
Sequences = List[Sequence]
TrainData = Tuple[Sequences, Sequences]

TrainedModel = namedtuple(
    "TrainedModel",
    ["model", "source_vocabulary", "target_vocabulary", "feature_vocabulary", "metrics", "checkpoint", "settings"]
)


def _prepare_datasets(settings: Settings, train_data: RawDataset, development_data: Optional[RawDataset],
                      autoregressive: bool = True, use_features: bool = False):
    # Build vocabularies
    source_vocabulary = SourceVocabulary.build_vocabulary(
        train_data.sources, min_frequency=settings.min_source_frequency
    )
    target_vocabulary = TransducerVocabulary.build_vocabulary(
        train_data.targets, min_frequency=settings.min_target_frequency
    )

    if use_features:
        feature_vocabulary = SourceVocabulary.build_vocabulary(train_data.features)
    else:
        feature_vocabulary = None

    dataset_class = AutoregressiveTransducerDatasetTrain if autoregressive else NonAutoregressiveTransducerDatasetTrain
    train_dataset = dataset_class(
        dataset=train_data, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
        feature_vocabulary=feature_vocabulary, use_features=use_features
    )

    if development_data is not None:
        development_dataset = dataset_class(
            dataset=development_data, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
            feature_vocabulary=feature_vocabulary, use_features=use_features
        )
    else:
        development_dataset = None

    return source_vocabulary, target_vocabulary, feature_vocabulary, train_dataset, development_dataset


def _build_model(source_vocab_size: int, target_vocab_size: int, settings: Settings,
                 feature_vocab_size: Optional[int] = None) -> TransducerModel:
    if settings.model == "autoregressive":
        return LSTMEncoderDecoderModel(
            source_vocab_size=source_vocab_size, target_vocab_size=target_vocab_size,
            embedding_dim=settings.embedding_size, hidden_size=settings.hidden_size,
            num_layers=settings.hidden_layers, dropout=settings.dropout, device=settings.device,
            scorer=settings.scorer, temperature=settings.temperature, use_features=settings.use_features,
            feature_vocab_size=feature_vocab_size, feature_encoder_hidden=settings.hidden_size,
            feature_encoder_pooling=settings.features_pooling, feature_encoder_layers=settings.features_num_layers,
            encoder_bridge=settings.encoder_bridge
        )

    elif settings.model == "non-autoregressive":
        return NonAutoregressiveLSTM(
            source_vocab_size=source_vocab_size, target_vocab_size=target_vocab_size,
            embedding_dim=settings.embedding_size, hidden_size=settings.hidden_size,
            num_layers=settings.hidden_layers, dropout=settings.dropout, device=settings.device,
            tau=settings.tau, scorer=settings.scorer, temperature=settings.temperature,
            use_features=settings.use_features, feature_vocab_size=feature_vocab_size,
            feature_encoder_layers=settings.features_num_layers, feature_encoder_hidden=settings.hidden_size,
            feature_encoder_pooling=settings.features_pooling, decoder_type=settings.non_autoregressive_decoder,
            max_targets_per_symbol=settings.max_targets_per_symbol
        )

    elif settings.model == "soft-attention":
        return SoftAttentionModel(
            source_vocab_size=source_vocab_size, target_vocab_size=target_vocab_size,
            embedding_dim=settings.embedding_size, hidden_size=settings.hidden_size,
            num_layers=settings.hidden_layers, dropout=settings.dropout, device=settings.device,
            scorer=settings.scorer, temperature=settings.temperature, use_features=settings.use_features,
            feature_vocab_size=feature_vocab_size, feature_encoder_hidden=settings.hidden_size,
            feature_encoder_pooling=settings.features_pooling, feature_encoder_layers=settings.features_num_layers,
            encoder_bridge=settings.encoder_bridge
        )

    else:
        raise ValueError(f"Unknown model type: {settings.model}")


def _build_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")


def _build_scheduler(optimizer, scheduler: str, gamma: float = 1.0, max_learning_rate: float = 0.1,
                     total_steps: int = None):
    if scheduler == "exponential":
        scheduler_model = ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == "one-cycle":
        scheduler_model = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, div_factor=100)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    def scheduler_step(epoch_end: bool):
        if scheduler == "exponential" and epoch_end:
            scheduler_model.step()
        elif scheduler == "one-cycle" and not epoch_end:
            scheduler_model.step()
        else:
            pass

    return scheduler_step


def _count_model_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def moving_avg_loss(old_loss: float, new_loss: float, gamma: float = 0.95) -> float:
    if old_loss is None:
        return new_loss
    else:
        return gamma * old_loss + (1 - gamma) * new_loss


def save_model(model: TrainedModel, name: str, path: str) -> str:
    os.makedirs(path, exist_ok=True)
    model_save_info = dict()
    model_save_info["model_class"] = type(model.model)
    model_save_info["parameters"] = model.model.get_params()
    model_save_info["state_dict"] = model.model.state_dict()
    model_save_info["source_vocabulary"] = model.source_vocabulary
    model_save_info["target_vocabulary"] = model.target_vocabulary
    model_save_info["feature_vocabulary"] = model.feature_vocabulary
    model_save_info["metrics"] = model.metrics
    model_save_info["checkpoint"] = model.checkpoint
    model_save_info["settings"] = model.settings

    save_model_path = os.path.join(path, name + ".pt")
    torch.save(model_save_info, save_model_path)

    return save_model_path


def load_model(path: str) -> TrainedModel:
    model_save_info = torch.load(path)

    model = model_save_info["model_class"](**model_save_info["parameters"])
    model.load_state_dict(model_save_info["state_dict"])

    source_vocabulary = model_save_info["source_vocabulary"]
    target_vocabulary = model_save_info["target_vocabulary"]
    feature_vocabulary = model_save_info["feature_vocabulary"]
    metrics = model_save_info["metrics"]
    checkpoint = model_save_info["checkpoint"]
    settings = model_save_info["settings"]

    return TrainedModel(
        model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
        feature_vocabulary=feature_vocabulary, metrics=metrics, checkpoint=checkpoint, settings=settings
    )


def evaluate_on_development_set(model_name: str, model: TrainedModel, development_data: TransducerDatasetTrain,
                                batch_size: int, fast_autoregressive_loss: bool, device: torch.device,
                                max_decoding_length: int):
    assert development_data is not None

    source_vocabulary = model.source_vocabulary
    target_vocabulary = model.target_vocabulary
    feature_vocabulary = model.feature_vocabulary
    model: TransducerModel = model.model

    model = model.eval()
    development_dataloader = DataLoader(
        development_data, batch_size=batch_size, shuffle=False, collate_fn=development_data.collate_fn
    )

    losses = []
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in development_dataloader:
            batch_sources = [
                [symbol for symbol in source if symbol not in source_vocabulary.get_specials()]
                for source in batch.raw_sources
            ]

            if model.use_features:
                batch_features = [
                    [symbol for symbol in feats if symbol not in feature_vocabulary.get_specials()]
                    for feats in batch.raw_features
                ]
            else:
                batch_features = None

            if model_name == "autoregressive":
                batch_losses = _get_autoregressive_loss(
                    model=model, batch=batch, device=device, allow_copy=True, enforce_copy=False, reduction='none',
                    fast=fast_autoregressive_loss
                )
                batch_predictions = autoregressive_greedy_sampling(
                    model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
                    sequences=batch_sources, max_decoding_length=max_decoding_length, features=batch_features,
                    feature_vocabulary=feature_vocabulary
                )

            elif model_name == "non-autoregressive":
                batch_losses = _get_non_autoregressive_loss(
                    model=model, batch=batch, device=device, allow_copy=True, enforce_copy=False, noop_discount=1.0,
                    reduction='none'
                )
                batch_predictions = non_autoregressive_inference(
                    model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
                    sequences=batch_sources, features=batch_features, feature_vocabulary=feature_vocabulary,
                    max_decoding_length=max_decoding_length
                )

            elif model_name == "soft-attention":
                batch_losses = _get_soft_attention_loss(model=model, batch=batch, reduction="none")
                batch_predictions = soft_attention_greedy_sampling(
                    model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
                    sequences=batch_sources, max_decoding_length=max_decoding_length, features=batch_features,
                    feature_vocabulary=feature_vocabulary
                )

            else:
                raise ValueError(f"Unknown model name: {model_name}")

            batch_losses = batch_losses.detach().cpu().tolist()
            losses.extend(batch_losses)

            batch_predictions = [prediction for prediction, _ in batch_predictions]
            batch_predictions = [
                [symbol for symbol in prediction if symbol not in target_vocabulary.get_special_symbols()]
                for prediction in batch_predictions
            ]
            predictions.extend(batch_predictions)

            batch_targets = [
                [symbol for symbol in target if symbol not in target_vocabulary.get_special_symbols()]
                for target in batch.raw_targets
            ]
            targets.extend(batch_targets)

    # Calculate metrics
    loss = np.mean(losses)
    wer = 100 * (1 - np.mean([prediction == target for prediction, target in zip(predictions, targets)]))
    edit_distance = np.mean(
        [editdistance.distance(prediction, target) for prediction, target in zip(predictions, targets)]
    )

    return {
        'loss': loss,
        'wer': wer,
        'edit_distance': edit_distance
    }


def _autoregressive_get_scores(model: TransducerModel, batch: Batch) -> Tensor:
    source_encodings = model.encode(batch.sources, batch.source_lengths)
    target_encodings, _ = model.decode(
        batch.targets, batch.target_lengths, source_encodings, batch.source_lengths
    )

    return model.get_transduction_scores(
        source_encodings, target_encodings, batch.features, batch.feature_lengths
    )


def _get_autoregressive_loss(model: TransducerModel, batch: Batch, device: torch.device, allow_copy: bool,
                             enforce_copy: bool, reduction: str, fast: bool) -> Tensor:
    scores = _autoregressive_get_scores(model=model, batch=batch)
    criterion = fast_autoregressive_transduction_loss if fast else autoregressive_transduction_loss

    loss = criterion(
        scores=scores, source_lengths=batch.source_lengths, target_lengths=batch.target_lengths,
        insertion_labels=batch.insertion_labels, substitution_labels=batch.substitution_labels,
        copy_index=batch.copy_index, copy_shift_index=batch.copy_shift_index, deletion_index=batch.deletion_index,
        copy_matrix=batch.copy_matrix, device=device, allow_copy=allow_copy, enforce_copy=enforce_copy,
        reduction=reduction
    )

    return loss


def _get_soft_attention_loss(model: TransducerModel, batch: Batch, reduction: str) -> Tensor:
    targets = batch.targets[:, 1:]
    target_lengths = batch.target_lengths - 1

    source_encodings = model.encode(batch.sources, batch.source_lengths)
    target_encodings, _ = model.decode(targets, target_lengths, source_encodings, batch.source_lengths)

    if model.use_features:
        scores = model.get_transduction_scores(target_encodings, batch.features, batch.feature_lengths)
    else:
        scores = model.get_transduction_scores(target_encodings)

    scores = scores[:, :-1]
    labels = batch.targets[:, 2:]  # Remove SOS symbols from labels
    loss = soft_attention_loss(scores=scores, target_labels=labels, reduction=reduction)
    return loss


def _non_autoregressive_get_scores(model: TransducerModel, batch: Batch) -> Tuple[Tensor, int]:
    tau = model.tau
    if tau is None:
        tau = batch.target_lengths.max().detach().cpu().item()

    if tau > model.max_targets_per_symbol:
        tau = model.max_targets_per_symbol

    if model.use_features:
        scores = model(
            sources=batch.sources, lengths=batch.source_lengths, features=batch.features,
            feature_lengths=batch.feature_lengths, tau=tau
        )
    else:
        scores = model(
            sources=batch.sources, lengths=batch.source_lengths, features=None, feature_lengths=None, tau=tau
        )

    return scores, tau


def _get_non_autoregressive_loss(model: TransducerModel, batch: Batch, device: torch.device, allow_copy: bool,
                                 enforce_copy: bool, noop_discount: float, reduction: str) -> Tensor:
    scores, tau = _non_autoregressive_get_scores(model=model, batch=batch)

    loss = non_autoregressive_transduction_loss(
        scores=scores, source_lengths=batch.source_lengths, target_lengths=batch.target_lengths,
        insertion_labels=batch.insertion_labels, substitution_labels=batch.substitution_labels,
        copy_index=batch.copy_index, copy_shift_index=batch.copy_shift_index,
        deletion_index=batch.deletion_index, noop_index=batch.noop_index,
        copy_matrix=batch.copy_matrix, device=device, allow_copy=allow_copy, enforce_copy=enforce_copy, tau=tau,
        noop_discount=noop_discount, reduction=reduction, return_backpointers=False
    )

    return loss


def train(train_data: RawDataset, development_data: Optional[RawDataset], settings: Settings) -> TrainedModel:
    if settings.verbose:
        logger.info("Prepare for Training")
        logger.info("Build vocabulary and datasets")

    is_non_autoregressive = settings.model == "non-autoregressive"
    is_autoregressive = not is_non_autoregressive

    source_vocabulary, target_vocabulary, feature_vocabulary, train_dataset, dev_dataset = _prepare_datasets(
        settings=settings, train_data=train_data, development_data=development_data,
        autoregressive=is_autoregressive, use_features=settings.use_features
    )
    max_development_decoding_length = max([len(datapoint.target) for datapoint in train_dataset]) + 10

    if settings.verbose:
        logger.info(f"Train data contains {len(train_dataset)} datapoints")
        if dev_dataset is not None:
            logger.info(f"Dev data contains {len(dev_dataset)} datapoints")
        logger.info(f"Source vocabulary contains {len(source_vocabulary)} items")
        logger.info(f"Target vocabulary contains {len(target_vocabulary)} actions")

    train_dataloader = DataLoader(
        train_dataset, batch_size=settings.batch, shuffle=True, collate_fn=train_dataset.collate_fn
    )

    if settings.verbose:
        logger.info("Build model")

    feature_vocab_size = None if feature_vocabulary is None else len(feature_vocabulary)
    if settings.model == "soft-attention":
        target_vocab_size = len(target_vocabulary.symbols)
    else:
        target_vocab_size = len(target_vocabulary)

    model = _build_model(
        source_vocab_size=len(source_vocabulary), target_vocab_size=target_vocab_size, settings=settings,
        feature_vocab_size=feature_vocab_size
    )

    if settings.verbose:
        num_model_parameters = _count_model_parameters(model)
        logger.info(f"Model has {num_model_parameters} parameters")
        logger.info(f"Device: {settings.device}")
    model = model.to(device=settings.device)
    model = model.train()

    if settings.verbose:
        logger.info("Build optimizer")
    optimizer = _build_optimizer(
        model=model, optimizer=settings.optimizer, lr=settings.lr, weight_decay=settings.weight_decay
    )

    if settings.verbose:
        logger.info("Build scheduler")

    scheduler_step = _build_scheduler(
        optimizer, scheduler=settings.scheduler, gamma=settings.gamma, max_learning_rate=settings.lr,
        total_steps=settings.epochs * len(train_dataloader)
    )

    if settings.verbose:
        logger.info("Start Training")

    running_loss = None
    step_counter = 0
    best_model_metric = np.inf
    best_checkpoint_path = None
    total_num_steps = settings.epochs * len(train_dataloader)

    for epoch in range(1, settings.epochs + 1):
        # Train epoch
        model = model.train()
        epoch_losses = []

        for batch in train_dataloader:
            optimizer.zero_grad()

            if settings.model == "autoregressive":
                loss = _get_autoregressive_loss(
                    model=model, batch=batch, allow_copy=settings.allow_copy, enforce_copy=settings.enforce_copy,
                    device=settings.device, reduction='mean', fast=settings.fast_autoregressive
                )

            elif settings.model == "non-autoregressive":
                loss = _get_non_autoregressive_loss(
                    model=model, batch=batch, allow_copy=settings.allow_copy, enforce_copy=settings.enforce_copy,
                    device=settings.device, noop_discount=settings.noop_discount, reduction='mean'
                )

            elif settings.model == "soft-attention":
                loss = _get_soft_attention_loss(model=model, batch=batch, reduction="mean")

            else:
                raise ValueError(f"Unknown model: {settings.model}")

            # Update parameters
            loss.backward()
            if settings.grad_clip is not None:
                clip_grad_value_(model.parameters(), settings.grad_clip)
            optimizer.step()
            scheduler_step(epoch_end=False)

            # Display loss
            step_counter += 1
            loss_item = loss.detach().cpu().item()
            running_loss = moving_avg_loss(running_loss, loss_item)
            epoch_losses.append(loss_item)

            if settings.verbose:
                if step_counter % settings.report_progress_every == 0 or step_counter == 1:
                    progress = 100 * step_counter / total_num_steps
                    current_learning_rate = optimizer.param_groups[0]['lr']
                    logger.info(
                        f"[{progress:.2f}%]" +
                        f" Loss: {running_loss:.3f}" +
                        f" || LR: {current_learning_rate:.6f}" +
                        f" || Step {step_counter} / {total_num_steps}"
                    )

        scheduler_step(epoch_end=True)

        # Evaluate on dev set
        epoch_model = TrainedModel(
            model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
            feature_vocabulary=feature_vocabulary, metrics=None, checkpoint=None, settings=settings
        )

        if dev_dataset is not None and (epoch % settings.evaluate_every == 0 or epoch == settings.epochs):
            development_metrics = evaluate_on_development_set(
                model_name=settings.model, model=epoch_model, development_data=dev_dataset,
                batch_size=settings.batch, device=settings.device, max_decoding_length=max_development_decoding_length,
                fast_autoregressive_loss=settings.fast_autoregressive
            )

            if settings.verbose:
                logger.info(
                    f"[Development metrics]    " +
                    f"Loss: {development_metrics['loss']:.4f}" +
                    f" || WER: {development_metrics['wer']:.2f}" +
                    f" || Edit-Distance: {development_metrics['edit_distance']:.2f}"
                )

        elif dev_dataset is None:
            development_metrics = None

        else:
            continue

        if development_metrics is not None:
            epoch_model_metric = development_metrics[settings.main_metric]
        else:
            epoch_model_metric = np.mean(epoch_losses)

        model_improved = epoch_model_metric < best_model_metric
        best_model_metric = epoch_model_metric if model_improved else best_model_metric
        save_metrics = development_metrics if development_metrics is not None else {'loss': np.mean(epoch_losses)}

        epoch_model = TrainedModel(
            model=model, source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary,
            feature_vocabulary=feature_vocabulary, metrics=save_metrics, checkpoint=epoch, settings=settings
        )

        if settings.keep_only_best_checkpoint:
            if model_improved or epoch == 1:
                if settings.verbose:
                    logger.info(f"Saving Model after epoch {epoch}")
                checkpoint_path = save_model(model=epoch_model, name=settings.name, path=settings.save_path)
            else:
                checkpoint_path = best_checkpoint_path
        else:
            if settings.verbose:
                logger.info(f"Saving Model after epoch {epoch}")
            checkpoint_path = save_model(model=epoch_model, name=settings.name + f"_{epoch}", path=settings.save_path)

        if model_improved or epoch == 1:
            best_checkpoint_path = checkpoint_path

    model = load_model(best_checkpoint_path)
    return model
