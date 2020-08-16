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
    def __init__(self, model, data_shape, num_classes, min_examples, max_examples, example_increment,
                 num_permutations=1, dataset=None, optimizer=None, lr_scheduler=False, training_params=None,
                 seed=None, synthetic_dtype="uint8", workers=8, gpu_dl=False, progress_bar=False):
        """
        Main class that is used for computing the empirical shattering dimensions of the network.
        Although the shattering dimension is defined only for binary class of functions, we treat the
        empirical shattering dimension as a dataset specific measure, taking both the number of
        classes as well as the size of the dataset into account.
        :param model: a callable that takes an input tensor and returns the model logits.
        :param data_shape: shape of the data -- this is specifically required when using synthetic data.
        :param num_classes: number of classes considered in the dataset.
        :param min_examples: minimum number of examples to be used for training the model. The model starts training
                                from example_increment to max_examples with increments of example_increment.
        :param max_examples: maximum number of examples to be used for training the model. The model starts training
                                from example_increment to max_examples with increments of example_increment.
        :param example_increment: number of examples to increment after each of training the model. The model starts
                                training from example_increment to max_examples with increments of example_increment.
        :param num_permutations (optional): number of permutations of the labels to be used for training the model for
                                a particular number of examples. This is a poor proxy in place of considering all
                                possible label permutations for which the actual shattering dimension is defined.
                                Should be kept as high as possible based on the availability of computational resources.
                                There is usually a high variance on the results based on the label permutations.
                                Picks the lowest accuracy among all the permutations to identify the final accuracy.
                                If not defined, defaults to 1.
        :param dataset: (optional) dataset object to be used for training the model (not required for synthetic datasets).
        :param optimizer: (optional) optimizer to be used. If not defined, defaults to adam with an LR of 1e-3.
        :param lr_scheduler: (optional) LR scheduler to be used. If not defined, defaults to a cosine LR schedule.
        :param training_params: (optional) provides params to the trainer which includes LR, train_epochs, momentum,
                                and weight decay.
        :param seed: (optional) seed value to be used for seeding the library.
        :param synthetic_dtype: (optional) Specifies the type of data to be used for training the model -- required
                                for synthetic data pipeline. Possible options are ["uint8", "float"]. Uint8 is a
                                reasonable option only for image dataset where each pixel takes on the distinct value
                                from 0-255. This significantly reduces the amount of memory required to store the
                                dataset. For using the library with non-image data, it's recommended to use float
                                as the datatype. If not defined, defaults to uint8.
        :param workers: (optional) number of workers to be used for training the model. If not defined, defaults to 8.
        :param gpu_dl: (optional) use GPU dataloader which internally uses prefetching to speed up the training. If
                                not defined, defaults to false.
        :param progress_bar: (optional) use progress bar to indicate training progress. If not defined, defaults to false.
        :return object of the class
        """
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

        self.min_examples = min_examples
        self.max_examples = max_examples
        self.example_increment = example_increment

        self.seed = seed
        self.workers = workers
        self.gpu_dl = gpu_dl
        self.progress_bar = progress_bar
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.dataset = dataset
        self.num_permutations = num_permutations

        # TODO: Add implementation for number of permutations, picking the minimum final accuracy
        if self.num_permutations > 0:
            raise NotImplementedError

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
        if training_params is None:  # Use standard params
            train_epochs = 50
            training_params = dict(optimizer="adam", lr=1e-3, train_epochs=train_epochs, wd=0.0, lr_scheduler="cosine")
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
        self.lr_scheduler = lr_scheduler

    def evaluate(self, acc_thresh=0.8, termination_thresh=0.1):
        """
        Evaluation method which computes the empirical shattering dimensions of the network.
        :param acc_thresh: (optional) defines the accuracy to be used as the cutoff point to identify the shattering
                            dimensions of the network. As the training process is noisy, this should be slightly
                            lower than perfect. If not defined, defaults to 0.8.
        :param termination_thresh: (optional) defines the accuracy at which to terminate the search. This is usually
                            kept to be very low. This should be a point where we are sure that we have already passed
                            the shattering dimensions of the network. If not defined, defaults to 0.1.
        :return a tuple of the computed shattering dimension as well as a dictionary containing detailed logs.
        """
        shattering_dims = {}
        for num_examples in range(self.min_examples, self.max_examples + self.example_increment, self.example_increment):
            logging_utils.log_info(f"Training model using {num_examples} examples...")

            # Reload model state and use random sampler to fix the number of examples in the dataset
            self.model.load_state_dict(self.initial_model_state)
            dataloader = dataloader_utils.get_dataloader(self.dataset, self.training_params["bs"], num_examples, workers=self.workers,
                                                         _worker_init_fn=self._worker_init_fn, world_size=self.world_size, gpu_dl=self.gpu_dl)
            
            if self.lr_scheduler is None:
                lr_scheduler = training_utils.get_lr_policy(self.model.optimizer, self.training_params["lr_scheduler"], self.training_params)
                logging_utils.log_info(f"LR scheduler: {lr_scheduler.__class__.__name__}")
                last_lr = lr_scheduler.get_last_lr()[0]
            else:
                lr_scheduler = self.lr_scheduler
                last_lr = self.training_params["lr"]

            for epoch in range(self.train_epochs):
                start = time.time()
                logging_utils.log_debug(f"Starting training for epoch # {epoch + 1} with an LR of {last_lr:.6f}!")
                training_utils.train(self.model, dataloader, device=self.device, progress_bar=self.progress_bar)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                    last_lr = lr_scheduler.get_last_lr()[0]
                logging_utils.log_debug(f"Training for {epoch + 1} finished. Time elapsed: {(time.time()-start)/60.:.4f} mins.")
            acc, log_dict = training_utils.evaluate(self.model, dataloader, device=self.device, progress_bar=self.progress_bar)
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
