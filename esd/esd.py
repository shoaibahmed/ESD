import copy
import time
import random
import numpy as np
import torch

from . import logging_utils
from . import training_utils
from . import dataset_utils
from . import dataloader_utils


class EmpiricalShatteringDimension:
    def __init__(self, model, data_shape, num_classes, dataset=None, optimizer=None, training_params=None, seed=None,
                 synthetic_dtype="uint8", max_examples=10000, example_increment=100, workers=8, gpu_dl=False, verbose=False):
        assert synthetic_dtype in ["uint8", "float"]

        self.distributed, self.world_size, self.num_gpus, self.rank = False, 1, 1, 0
        if torch.distributed.is_initialized():
            self.distributed = True
            self.world_size = torch.distributed.get_world_size()
            self.num_gpus = torch.cuda.device_count()
            self.rank = torch.distributed.get_rank()
            logging_utils.log_info(f"Distributed environment detected with a world-size of {self.num_gpus} and {self.num_gpus} GPUs per node!")

        self._worker_init_fn = None
        if seed is not None:
            print("Using seed = {}".format(seed))
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed=seed)
            random.seed(seed)

            def _worker_init_fn(id):
                np.random.seed(seed=seed + id)
                random.seed(seed + id)
            self._worker_init_fn = _worker_init_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.initial_model_state = copy.deepcopy(model.state_dict())
        # dist_utils.broadcast_from_main(self.initial_model_state, is_tensor=False)  # DDP will take care of this

        self.seed = seed
        self.workers = workers
        self.gpu_dl = gpu_dl
        self.verbose = verbose
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.max_examples = max_examples
        self.ex_inc = example_increment
        self.dataset = dataset

        if self.dataset is None:
            logging_utils.log_info(f"Initializing synthetic dataset with {self.max_examples} {synthetic_dtype} examples and {self.num_classes} number of classes!")
            self.dataset = dataset_utils.get_syntetic_dataset(self.max_examples, self.data_shape, self.num_classes,
                                                              dtype=synthetic_dtype, world_size=self.world_size, gpu_dl=self.gpu_dl)
        else:
            # Ensure same targets are generated at each process
            logging_utils.log_info("Replacing dataset targets with random targets!")
            assert isinstance(self.dataset.targets, list) or len(self.dataset.targets.shape) == 1
            dataset_utils.replace_dataset_targets(self.dataset, num_classes)

        # Define the optimizer
        if training_params is None:
            training_params = dict(optimizer="adam", lr=1e-3, train_epochs=50, wd=0.0)  # Use standard params
        else:
            self.training_params = training_params
        self.train_epochs = training_params["train_epochs"]
        if optimizer is None:
            if training_params["optimizer"] == "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=training_params["lr"],
                                             weight_decay=training_params["wd"])
            else:
                optimizer = torch.optim.SGD(self.model.parameters(), lr=training_params["lr"],
                                            momentum=training_params["momentum"], weight_decay=training_params["wd"])

        assert optimizer is not None
        self.model.optimizer = optimizer
        self.model.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, acc_thresh=0.8, termination_thresh=0.1):
        shattering_dims = {}
        for num_examples in range(self.ex_inc, self.max_examples + self.ex_inc, self.ex_inc):
            logging_utils.log_info(f"Training model using {num_examples} examples...")

            # Reload model state and use random sampler to fix the number of examples in the dataset
            self.model.load_state_dict(self.initial_model_state)
            dataloader = dataloader_utils.get_dataloader(self.dataset, self.training_params["bs"], num_examples, workers=self.workers,
                                                         _worker_init_fn=self._worker_init_fn, world_size=self.world_size, gpu_dl=self.gpu_dl)

            for epoch in range(self.train_epochs):
                start = time.time()
                logging_utils.log_debug(f"Starting training for epoch # {epoch + 1}...")
                training_utils.train(self.model, dataloader, device=self.device, logging=self.verbose)
                logging_utils.log_debug(f"Training for {epoch + 1} finished. Time elapsed: {(time.time()-start)/60.} mins.")
            acc, log_dict = training_utils.evaluate(self.model, dataloader, device=self.device, logging=self.verbose)
            assert log_dict['total'] == num_examples, f"{log_dict['total']} != {num_examples}"
            shattering_dims[num_examples] = acc
            if acc < termination_thresh:
                logging_utils.log_info(f"Terminating ESD search after encountering an accuracy less than the termination thresh ({acc} < {termination_thresh}) at {num_examples} examples!")
                break

        shattering_dim = -1
        num_examples = sorted(list(shattering_dims.keys()))
        for num_example in num_examples:
            acc = shattering_dims[num_example]
            if acc >= acc_thresh:
                shattering_dim = num_example
        return shattering_dim, shattering_dims
