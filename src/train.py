import os
import time
import uuid

import numpy as np
import pandas as pd
import torch
import copy

import torch.nn.functional as F
import egg.core as core

from datetime import timedelta

from .models import Sender, DiscriReceiver, Vision
from .dataloader import get_datasets, get_loaders
from .utils import get_params
from .logger import logger
from .evaluation import CommunicationMetricsCallback
from .pretraining import pretraining


def main(args, experiment=None, device=None):
    """
    Main function to run the training
    :param args: list of arguments
    :param device: torch.device, device to run the training on
    :return:
    """
    run_uuid = uuid.uuid4()

    logger.title(f"Starting Agent-based motherese training run {run_uuid}")

    opts = core.init(get_params(args), args)
    opts.run_uuid = run_uuid

    logger.debug(f"opts in main: {opts}")

    opts_dict = vars(opts)

    logger.table(opts_dict, title="Parameters")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"))
        logger.info(f"device: {device}")
        if device == "mps":
            torch.mps.empty_cache()

    train_dataset, test_dataset, vision_train_dataset, vision_test_dataset = get_datasets(opts)
    train_loader, test_loader, vision_train_loader, vision_test_loader = get_loaders(opts, train_dataset, test_dataset, vision_train_dataset, vision_test_dataset)

    logger.info("dataloaders initialised")

    def loss(
        _sender_input,
        _message,
        _receiver_input,
        receiver_output,
        labels,
        _aux_input,
    ):
        acc = (receiver_output.argmax(-1) == labels).float()
        loss_result = F.cross_entropy(receiver_output, labels.long(), reduction="none")

        aux_info = {
            "acc": acc,
            "loss": loss_result,
        }

        return loss_result, aux_info

    n_features = opts.image_width * opts.image_height * 3
    logger.info(f"number of features per image (including colour channels): {n_features}")

    logger.info("initialising Vision module")
    vision = pretraining(
        Vision(
            image_size=(opts.image_width, opts.image_height),
            n_hidden=opts.sender_hidden,
            n_classes=train_dataset.get_n_classes(),
        ),
        device,
        vision_train_loader,
        vision_test_loader,
        opts=opts,
    )

    if opts.vision_from_file != '':
        vision.load_state_dict(torch.load(opts.vision_from_file, weights_only=True))
        vision.eval()
        logger.info(f"Loaded pretrained vision module from {opts.vision_from_file}")

    logger.info(f"Vision module pretrained")

    st = time.time()

    sender = core.RnnSenderGS(
        Sender(
            vision,
            opts.sender_hidden,
        ),
        vocab_size=opts.vocab_size,
        embed_dim=opts.sender_embedding,
        hidden_size=opts.sender_hidden,
        max_len=opts.max_len,
        trainable_temperature=opts.trainable_temperature,
        temperature=3.0,
        cell=opts.sender_cell
    )

    receiver = core.RnnReceiverGS(
        DiscriReceiver(
            n_hidden=opts.receiver_hidden,
            image_size=(opts.image_width, opts.image_height)
        ),
        vocab_size=opts.vocab_size,
        embed_dim=opts.receiver_embedding,
        hidden_size=opts.receiver_hidden,
        cell=opts.receiver_cell
    )

    logger.info("Setup game")

    game = core.SenderReceiverRnnGS(
        sender,
        receiver,
        loss,
    )

    metric_plot_callback = CommunicationMetricsCallback(
        opts=opts,
        device=device,
    )
    metric_plot_callback.set_uuid(run_uuid)

    callbacks = [
        core.ConsoleLogger(as_json=True, print_train_loss=True),
        metric_plot_callback,
        core.TemperatureUpdater(
            agent=sender,
            decay=0.7
        ),
    ]

    optimizer = core.build_optimizer(game.parameters())

    trainer = core.Trainer(
        game=game,
        optimizer=optimizer,
        train_data=train_loader,
        validation_data=test_loader,
        callbacks=callbacks,
    )

    logger.debug("start training: ")
    durations = []

    trainer.train(opts.n_epochs)
    durations.append(time.time() - st)

    logger.info(f"Round 0 finished in {timedelta(seconds=time.time() - st)} seconds")

    logger.info("Training finished")

    control_group_saved_sender_state = sender.state_dict()
    control_group_saved_receiver_state = receiver.state_dict()

    control_group_metric_plot_callback = copy.deepcopy(metric_plot_callback)
    control_group_metric_plot_callback.set_control_group()

    for round in range(1, opts.rounds):
        st = time.time()
        logger.info(f"starting round {round} / {opts.rounds - 1}")
        logger.info("initialising Sender module")
        saved_sender_state = sender.state_dict()
        sender = core.RnnSenderGS(
            Sender(
                vision,
                opts.sender_hidden,
            ),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            temperature=3.0,
            trainable_temperature=opts.trainable_temperature,
            cell=opts.sender_cell
        )
        sender.load_state_dict(saved_sender_state)

        receiver = core.RnnReceiverGS(
            DiscriReceiver(
                n_hidden=opts.receiver_hidden,
                image_size=(opts.image_width, opts.image_height)
            ),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell
        )

        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            metric_plot_callback,
            core.TemperatureUpdater(
                agent=sender,
                decay=0.7
            ),
        ]

        game = core.SenderReceiverRnnGS(
            sender,
            receiver,
            loss,
        )

        optimizer = core.build_optimizer(game.parameters())

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks,
        )

        trainer.train(opts.n_epochs)

        logger.info(f"Round {round} finished in {(time.time() - st).__round__(2)} seconds")
        experiment_group_rounds_left = int(opts.rounds) - round - 1
        logger.debug(f"rounds left for current experiment: {experiment_group_rounds_left}")
        control_group_rounds_left = int(opts.rounds) - 1
        logger.debug(f"rounds left for control group: {control_group_rounds_left}")
        total_rounds_left = experiment_group_rounds_left + control_group_rounds_left
        logger.debug(f"total rounds left: {total_rounds_left}")
        average_run_duration = np.average(durations)
        logger.info(f"estimated time remaining for current experiment: {timedelta(seconds=(total_rounds_left * average_run_duration))}")
        total_duration_current_experiment = (2 * (opts.rounds - 1) + 1) * average_run_duration
        logger.info(f"estimated time remaining for all experiments: {timedelta(seconds=(total_duration_current_experiment * (opts.n_experiments - experiment) + (total_rounds_left * average_run_duration) ))}")

        logger.info("Training finished")
        # logger.info(f"running control group for {opts.n_epochs * (opts.rounds - 1)} epochs")

    for round in range(1, opts.rounds):
        st = time.time()
        logger.info(f"starting round {round} / {opts.rounds - 1}")
        logger.info("initialising Sender module")
        sender = core.RnnSenderGS(
            Sender(
                vision,
                opts.sender_hidden,
            ),
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            max_len=opts.max_len,
            temperature=3.0,
            cell=opts.sender_cell
        )

        sender.load_state_dict(control_group_saved_sender_state)

        receiver = core.RnnReceiverGS(
            DiscriReceiver(
                n_hidden=opts.receiver_hidden,
                image_size=(opts.image_width, opts.image_height)
            ),
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell
        )
        receiver.load_state_dict(control_group_saved_receiver_state)

        callbacks=[
            core.ConsoleLogger(as_json=True, print_train_loss=True),
            control_group_metric_plot_callback,
            core.TemperatureUpdater(
                agent=sender,
                decay=0.7
            ),
        ]


        game = core.SenderReceiverRnnGS(
            sender,
            receiver,
            loss,
        )

        optimizer = core.build_optimizer(game.parameters())

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks,
        )

        trainer.train(opts.n_epochs)

        logger.info(f"Round {round} finished in {(time.time() - st).__round__(2)} seconds")
        experiment_group_rounds_left = int(opts.rounds) - round - 1
        logger.debug(f"rounds left for current experiment: {experiment_group_rounds_left}")
        control_group_rounds_left = int(opts.rounds) - 1
        logger.debug(f"rounds left for control group: {control_group_rounds_left}")
        total_rounds_left = experiment_group_rounds_left + control_group_rounds_left
        logger.debug(f"total rounds left: {total_rounds_left}")
        average_run_duration = np.average(durations)
        logger.info(f"estimated time remaining for current experiment: {timedelta(seconds=(total_rounds_left * average_run_duration))}")
        total_duration_current_experiment = (2 * (opts.rounds - 1) + 1) * average_run_duration
        logger.info(f"estimated time remaining for all experiments: {timedelta(seconds=(total_duration_current_experiment * (opts.n_experiments - experiment) + (total_rounds_left * average_run_duration) ))}")
        control_group_saved_sender_state = sender.state_dict()
        control_group_saved_receiver_state = receiver.state_dict()



    # sender = core.RnnSenderGS(
    #     Sender(
    #         vision,
    #         opts.sender_hidden,
    #     ),
    #     vocab_size=opts.vocab_size,
    #     embed_dim=opts.sender_embedding,
    #     hidden_size=opts.sender_hidden,
    #     max_len=opts.max_len,
    #     temperature=3.0,
    #     cell=opts.sender_cell
    # )
    # sender.load_state_dict(control_group_saved_sender_state)
    #
    # receiver = core.RnnReceiverGS(
    #     DiscriReceiver(
    #         n_hidden=opts.receiver_hidden,
    #         image_size=(opts.image_width, opts.image_height)
    #     ),
    #     vocab_size=opts.vocab_size,
    #     embed_dim=opts.receiver_embedding,
    #     hidden_size=opts.receiver_hidden,
    #     cell=opts.receiver_cell
    # )
    # receiver.load_state_dict(control_group_saved_receiver_state)
    #
    # callbacks=[
    #         core.ConsoleLogger(as_json=True, print_train_loss=True),
    #         control_group_metric_plot_callback,
    #         core.TemperatureUpdater(
    #             agent=sender,
    #             decay=0.7
    #         ),
    #     ]
    #
    # game = core.SenderReceiverRnnGS(
    #     sender,
    #     receiver,
    #     loss,
    # )
    #
    # optimizer = core.build_optimizer(game.parameters())
    #
    # trainer = core.Trainer(
    #     game=game,
    #     optimizer=optimizer,
    #     train_data=train_loader,
    #     validation_data=test_loader,
    #     callbacks=callbacks,
    # )
    #
    # trainer.train(opts.n_epochs * (opts.rounds-1))

    logger.info("Control group training finished")

    return opts.run_uuid


if __name__ == "__main__":
    import sys

    parser = get_params(sys.argv[1:])
    opts, _ = parser.parse_known_args()

    experiment_uuid = opts.experiment_uuid or uuid.uuid4()
    n_experiments = opts.n_experiments

    logger.title(f"Starting {n_experiments} experiments with uuid: {experiment_uuid}")

    runs = []
    run_dfs = []

    for i in range(1, n_experiments + 1):
        run = main(sys.argv[1:], experiment=i)
        logger.info(f"Run {i}/{n_experiments} finished")
        runs.append(run)
        run_dfs.append(pd.read_csv(f"./src/data/logs/{run}/metrics_data.csv"))

    full_df = pd.concat(
        run_dfs,
        ignore_index=True
    )
    full_df["experiment_uuid"] = experiment_uuid

    full_df_columns = pd.read_csv(f"./src/data/logs/{runs[0]}/metrics_data.csv").columns

    path = f"./src/data/experiments/{experiment_uuid}"
    os.makedirs(path, exist_ok=True)

    full_df.to_csv(f"{path}/full_metrics_data.csv", index=False)

