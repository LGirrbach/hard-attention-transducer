import os
import torch
import argparse

from typing import Optional
from collections import namedtuple

setting_fields = [
    # Training Settings
    "epochs",  # Number of epochs to train
    "batch",  # Batch size
    "device",  # Whether to use cuda
    "scheduler",  # Name of learning rate scheduler
    "gamma",  # Decay factor for exponential learning rate scheduler
    "verbose",  # Whether to print training progress
    "report_progress_every",  # Number of updates between loss reports
    "evaluate_every",  # Number of epochs between evaluating on dev set
    "main_metric",  # Development metric used for evaluating training progress (training loss if no dev data)
    "keep_only_best_checkpoint",  # Whether to only save the best checkpoint (according to loss / dev score)
    "use_features",  # Whether to use additional features (e.g. for inflection)
    "min_source_frequency",  # Mask source symbols by UNK token that appear less than given frequency
    "min_target_frequency",  # Mask target symbols by UNK token that appear less than given frequency

    # Optimizer Settings
    "optimizer",  # Name of optimizer
    "lr",  # (Initial / Max.) learning rate
    "weight_decay",  # Weight decay factor
    "grad_clip",  # Max. absolute value of gradients (not applied if None)

    # Model Settings
    "model",  # Model type (autoregressive, non-autoregressive, soft-attention)
    "embedding_size",  # Num dimensions of embedding vectors
    "hidden_size",  # Hidden size
    "hidden_layers",  # Num of layers of encoders / decoders
    "dropout",  # Dropout probability
    "tau",  # Branching factor for non-autoregressive model
    "non_autoregressive_decoder",  # Whether to use fixed / position / lstm decoder for non-autoregressive model
    "max_targets_per_symbol",  # Maximum number of target symbol decoded from a single input symbol
    "scorer",  # Normalisation function for raw probability scores
    "temperature",  # Scaling factor for raw prediction scores
    "features_num_layers",  # Number of hidden layers in feature encoder
    "features_pooling",  # Pooling method to combine feature encodings
    "encoder_bridge",  # Pass summary of source sequence to autoregressive decoder

    # Loss Settings
    "noop_discount",  # Discount factor for loss incurred by blank actions (only for non-autoregressive model)
    "allow_copy",  # Whether copying is a valid action
    "enforce_copy",  # Always copy when possible (instead of substitution),
    "fast_autoregressive",  # Use fast autoregressive loss. In this case, copying the same symbol multiple times is not
                            # possible

    # Experiment Settings
    "name",  # Name of experiment
    "train_data_path",  # Path to train data
    "dev_data_path",  # Path to dev data
    "save_path",  # Where to save model checkpoints and settings

    # Inference Settings
    "beam_search",  # Whether to use beam search decoding for autoregressive transducers
    "num_beams",  # Number of beams for beam search decoding
    "max_decoding_length",  # Maximum length of decoded sequences (only autoregressive models)
]

Settings = namedtuple("Settings", field_names=setting_fields)


def save_settings(settings: Settings) -> None:
    os.makedirs(settings.save_path, exist_ok=True)
    with open(os.path.join(settings.save_path, f"{settings.name}_settings.tsv"), 'w') as ssf:
        for setting, value in settings._asdict().items():
            ssf.write(f"{setting}\t{value}\n")


def make_argument_parser():
    parser = argparse.ArgumentParser("Hard Attention Transducer Argument Parser")

    # Training Settings
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--cuda", action='store_true', help="Use GPU")
    parser.add_argument(
        "--scheduler", type=str, default="exponential", choices=["exponential", "one-cycle"], help="LR scheduler"
    )
    parser.add_argument("--gamma", type=float, default=1.0, help="Decay for Exponential LR Scheduler")
    parser.add_argument("--silent", action="store_true", help="Disables training progress output")
    parser.add_argument("--report-progress-every", type=int, default=10, help="Number of updates between loss reports")
    parser.add_argument("--evaluate-every", type=int, default=1, help="Number of epochs between evaluating on dev set")
    parser.add_argument(
        "--main-metric", type=str, choices=["loss", "wer", "edit_distance"], default="loss",
        help="Development metric used for evaluating training progress (training loss if no dev data)"
    )
    parser.add_argument(
        "--keep-only-best-checkpoint", action="store_true",
        help="Whether to only save the best checkpoint (according to loss / dev score)"
    )
    parser.add_argument("--use-features", action="store_true", help="Whether to use additional features")
    parser.add_argument(
        "--min-source-frequency", type=int, default=1,
        help="Mask source symbols by UNK token that appear less than given frequency"
    )
    parser.add_argument(
        "--min-target-frequency", type=int, default=1,
        help="Mask target symbols by UNK token that appear less than given frequency"
    )

    # Optimizer Settings
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight Decay")
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping value")

    # Model Settings
    parser.add_argument("--embedding", type=int, default=128, help="Embedding size")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--layers", type=int, default=1, help="Number of Encoder / Decoder Layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--tau", type=int, default=5, help="\\tau parameter for non-autoregressive model")
    parser.add_argument(
        "--non-autoregressive-decoder", type=str, default="position", choices=["fixed", "position", "lstm"],
        help="Type of decoder in non-autoregressive model"
    )
    parser.add_argument(
        "--max-targets-per-symbol", type=int, default=50,
        help="Max. target symbols that can be generated from 1 input symbol"
    )
    parser.add_argument(
        "--scorer", action='store', default="softmax", choices=["softmax", "entmax", "sparsemax"],
        help="Probability normalisation function"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Scaling factor for prediction scores (only used for training)"
    )
    parser.add_argument("--features-num-layers", type=int, default=0, help="Num layers in feature encoder")
    parser.add_argument(
        "--features-pooling", type=str, default="max", choices=["max", "sum", "mean", "dot", "mlp"],
        help="Pooling method for feature encoder"
    )
    parser.add_argument(
        "--encoder-bridge", action="store_true", help="Pass summary of encoder sequence to autoregressive decoder"
    )

    # Loss Settings
    parser.add_argument(
        "--noop-discount", type=float, default=1.0,
        help=(
            "Weight for 'no operation' action. Can be used to bias non-autoregressive model towards " +
            "copy/substitution operations."
        )
    )
    parser.add_argument("--disable-copy", action="store_true", help="Disables copy action")
    parser.add_argument("--enforce-copy", action="store_true", help="Always copy if possible")
    parser.add_argument(
        "--fast-autoregressive", action="store_true",
        help="Use fast autoregressive loss. In this case, copying the same symbol multiple times is not possible."
    )

    # Experiment Settings
    parser.add_argument("--name", type=str, default="trial_model", help="Experiment Name")
    parser.add_argument("--train-data", type=str, default=None, help="Path to train data")
    parser.add_argument("--dev-data", type=str, default=None, help="Path to dev data")
    parser.add_argument("--save-path", type=str, default="./saved_models", help="Where to save models")

    # Inference Settings
    parser.add_argument(
        "--beam-search", action="store_true", help="Enables beam search decoding for autoregressive model"
    )
    parser.add_argument("--beams", type=int, default=5, help="Number of beams in beam search decoding")
    parser.add_argument(
        "--max-decoding-length", type=int, default=70, help="Maximum length of decoded sequences (only autoregressive)"
    )

    return parser


def get_settings_from_arguments() -> Settings:
    argument_parser = make_argument_parser()
    args = argument_parser.parse_args()

    # Process arguments
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("Requested GPU, but no GPUs available")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    verbose = not args.silent
    allow_copy = not args.disable_copy

    # Instantiate settings
    settings = Settings(
        epochs=args.epochs, batch=args.batch, device=device, scheduler=args.scheduler, gamma=args.gamma,
        verbose=verbose, report_progress_every=args.report_progress_every, main_metric=args.main_metric,
        keep_only_best_checkpoint=args.keep_only_best_checkpoint, use_features=args.use_features,
        optimizer=args.optimizer, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip, model="lstm",
        embedding_size=args.embedding, hidden_size=args.hidden, hidden_layers=args.layers, dropout=args.dropout,
        tau=args.tau, non_autoregressive_decoder=args.non_autoregressive_decoder,
        max_targets_per_symbol=args.max_targets_per_symbol, scorer=args.scorer, temperature=args.temperature,
        features_num_layers=args.features_num_layers, features_pooling=args.features_pooling,
        noop_discount=args.noop_discount, allow_copy=allow_copy, enforce_copy=args.enforce_copy, name=args.name,
        train_data_path=args.train_data, dev_data_path=args.dev_data, save_path=args.save_path,
        beam_search=args.beam_search, num_beams=args.beams, max_decoding_length=args.max_decoding_length,
        encoder_bridge=args.encoder_bridge, evaluate_every=args.evaluate_every,
        min_source_frequency=args.min_source_frequency, min_target_frequency=args.min_target_frequency,
        fast_autoregressive=args.fast_autregressive
    )

    return settings


def make_settings(
        use_features: bool, model: str, name: str, save_path: str,
        epochs: int = 1, batch: int = 16, device: torch.device = torch.device('cpu'),
        scheduler: str = "exponential", gamma: float = 1.0, verbose: bool = True, report_progress_every: int = 10,
        main_metric: str = "loss", keep_only_best_checkpoint: bool = True, optimizer: str = "sgd", lr: float = 0.001,
        weight_decay: float = 0.0, grad_clip: Optional[float] = None, embedding_size: int = 128, hidden_size: int = 128,
        hidden_layers: int = 1, dropout: float = 0.0, tau: Optional[int] = 5, evaluate_every: int = 1,
        non_autoregressive_decoder: str = "position", max_targets_per_symbol: int = 50, scorer: str = "softmax",
        temperature: float = 1.0, features_num_layers: int = 0, features_pooling: str = "mean",
        noop_discount: float = 1.0, allow_copy: bool = True, enforce_copy: bool = False,
        train_data_path: Optional[str] = None, dev_data_path: Optional[str] = None, beam_search: bool = True,
        num_beams: int = 5, max_decoding_length: int = 100, encoder_bridge: bool = False,
        min_source_frequency: int = 1, min_target_frequency: int = 1, fast_autoregressive: bool = False) -> Settings:
    return Settings(
        epochs=epochs, batch=batch, device=device, scheduler=scheduler, gamma=gamma,
        verbose=verbose, report_progress_every=report_progress_every, main_metric=main_metric,
        keep_only_best_checkpoint=keep_only_best_checkpoint, use_features=use_features,
        optimizer=optimizer, lr=lr, weight_decay=weight_decay, grad_clip=grad_clip, model=model,
        embedding_size=embedding_size, hidden_size=hidden_size, hidden_layers=hidden_layers, dropout=dropout,
        tau=tau, non_autoregressive_decoder=non_autoregressive_decoder, max_targets_per_symbol=max_targets_per_symbol,
        scorer=scorer, temperature=temperature, features_num_layers=features_num_layers,
        features_pooling=features_pooling, noop_discount=noop_discount, allow_copy=allow_copy,
        enforce_copy=enforce_copy, name=name, train_data_path=train_data_path, dev_data_path=dev_data_path,
        save_path=save_path, beam_search=beam_search, num_beams=num_beams, max_decoding_length=max_decoding_length,
        encoder_bridge=encoder_bridge, evaluate_every=evaluate_every, fast_autoregressive=fast_autoregressive,
        min_source_frequency=min_source_frequency, min_target_frequency=min_target_frequency
    )
