import copy
import random
import numpy as np
import torch

from . import training_utils
from . import dataset_utils


class EmpiricalShatteringDimension:
    def __init__(self, model, data_shape, num_classes, dataset=None, synthetic_data=True, optimizer=None,
                 training_params=None, seed=None, max_examles=10000, example_increment=100, verbose=False):
        # Assertion tests
        # assert all([True])

        if seed is not None:
            print("Using seed = {}".format(args.seed))
            torch.manual_seed(args.seed + args.local_rank)
            torch.cuda.manual_seed(args.seed + args.local_rank)
            np.random.seed(seed=args.seed + args.local_rank)
            random.seed(args.seed + args.local_rank)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.initial_model_state = copy.deepcopy(model.state_dict())

        self.seed = seed
        self.verbose = verbose
        self.data_shape = data_shape
        self.num_classes = num_classes
        self.max_examples = max_examles
        self.ex_inc = example_increment
        self.dataset = dataset
        if self.dataset is None:
            self.dataset = dataset_utils.get_syntetic_dataset(self.max_examples, self.data_shape, self.num_classes)

        # Define the optimizer
        if training_params is None:
            training_params = dict(optimizer="adam", lr=1e-3, train_epochs=10, wd=0.0)  # Use standard params
        else:
            self.training_params = training_params
        self.train_epochs = training_params["train_epochs"]
        if optimizer is None:
            if training_params["optimizer"] == "adam":
                optimizer = torch.optim.Adam(self.model.parameters(), lr=training_params["lr"],
                                             weight_decay=training_params["wd"])
            else:
                optimizer =  torch.optim.SGD(self.model.parameters(), lr=training_params["lr"],
                                             momentum=training_params["momentum"], weight_decay=training_params["wd"])

        assert optimizer is not None
        self.model.optimizer = optimizer
        self.model.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, acc_thresh=0.8):
        shattering_dims = {}
        for num_examples in range(self.ex_inc, self.max_examples + self.ex_inc, self.ex_inc):
            print(f"[ESD] Training model using {num_examples} examples...")

            # Reload model state and use random sampler to fix the number of examples in the dataset
            self.model.load_state_dict(self.initial_model_state)
            dataloader = dataset_utils.get_dataloader(self.dataset, self.training_params["bs"], num_examples)

            for _ in range(self.train_epochs):
                training_utils.train(self.model, dataloader, device=self.device, logging=self.verbose)
            acc = training_utils.evaluate(self.model, dataloader, device=self.device, logging=self.verbose)
            shattering_dims[num_examples] = acc

        shattering_dim = -1
        num_examples = sort(list(shattering_dims.keys()))
        for num_example in num_examples:
            acc = shattering_dims[num_example]
            if acc >= self.acc_thresh:
                shattering_dim = num_example
        return shattering_dim, shattering_dims
