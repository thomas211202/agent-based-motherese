import argparse
import random

from src.logger import logger

def get_params(params):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of games to play (default: 3)",
    )

    parser.add_argument(
        "--dev",
        type=bool,
        default=False,
        help="If this flag is passed, the script runs in development mode (default: False)",
    )

    parser.add_argument(
        "--prod",
        type=bool,
        default=False,
        help="If this flag is passed, the script runs in development mode (default: False)",
    )

    parser.add_argument(
        "--n_distractors",
        type=int,
        default=10,
        help="Number of distractors in the discrimination game (default: 10)",
    )

    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers for the data loader (default: 4)",
    )

    parser.add_argument(
        "--pretrain_data_split",
        type=float,
        default=0.2,
        help="Fraction of the data to be used for pretraining (default: 0.2)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 1000000),
        help="Seed for the random number generator (default: 42)",
    )

    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Number of epochs for pretraining (default: 0)",
    )

    parser.add_argument(
        "--image_width",
        type=int,
        default=28,
        help="Width of the image (default: 28)",
    )

    parser.add_argument(
        "--image_height",
        type=int,
        default=28,
        help="Height of the image (default: 28)",
    )

    parser.add_argument(
        "--game_type",
        type=str,
        default="discri",
        help="Selects whether to play a reco(nstruction) or discri(mination) game (default: discri)",
    )
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    # (the following is only used in the reco game)
    parser.add_argument(
        "--n_attributes",
        type=int,
        default=None,
        help="Number of attributes in Sender input (must match data set, and it is only used in reco game)",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        default=None,
        help="Number of values for each attribute (must match data set)",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: gs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )

    parser.add_argument(
        "--receiver_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Receiver, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="lstm",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="lstm",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )

    parser.add_argument(
        "--vision_from_file",
        type=str,
        default='',
        help="Location to load vision module [not loading by default] (default: '')",
    )

    parser.add_argument(
        "--vision_to_file",
        type=str,
        default='',
        help="Location to store vision module [not storing by default] (default: '')",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="dataset (default: mnist)",
    )

    parser.add_argument(
        "--trainable_temperature",
        type=bool,
        default=False,
        help="trainable temperature for Gumbel-softmax (default: False)",
    )

    parser.add_argument(
        "--n_experiments",
        type=int,
        default=30,
        help="Number of experiments to run (default: 30)",
    )

    parser.add_argument(
        "--experiment_uuid",
        type=str,
        default=None,
        help="UUID of the experiment (default: None)",
    )

    return parser


def log_hyperparams(opts):
    logger.title("Hyperparameters")
    logger.info(f"game_type: {opts.game_type}")
    logger.info(f"batch size: {opts.batch_size}")
    logger.info(f"pretrain data split: {opts.pretrain_data_split}")
    logger.info(f"seed: {opts.seed}")
    logger.info(f"pretrain epochs: {opts.pretrain_epochs}")
    logger.info(f"n epochs: {opts.n_epochs}")
    logger.info(f"image size: {opts.image_width} x {opts.image_height}")
    logger.info(f"vocab size: {opts.vocab_size}")
    logger.info(f"max len: {opts.max_len}")
    logger.info(f"sender hidden: {opts.sender_hidden}")
    logger.info(f"receiver hidden: {opts.receiver_hidden}")
    logger.info(f"sender embedding: {opts.sender_embedding}")
    logger.info(f"receiver embedding: {opts.receiver_embedding}")
    logger.info(f"sender cell: {opts.sender_cell}")
    logger.info(f"receiver cell: {opts.receiver_cell}")
    logger.info(f"sender entropy coeff: {opts.sender_entropy_coeff}")
    logger.info(f"receiver entropy coeff: {opts.receiver_entropy_coeff}")
    logger.info(f"learning rate: {opts.lr}")
    logger.title("")
    return True
