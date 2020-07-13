import random
import numpy as np
import torch

from . import training_utils
from . import dataset


class EmpiricalShatteringDimension:
    def __init__(self, model, data_shape, num_classes, dataloader=None, synthetic_data=True, optimizer=None,
                 training_params=None, seed=None):
        # Assertion tests
        assert all([True])

        if seed is not None:
            print("Using seed = {}".format(args.seed))
            torch.manual_seed(args.seed + args.local_rank)
            torch.cuda.manual_seed(args.seed + args.local_rank)
            np.random.seed(seed=args.seed + args.local_rank)
            random.seed(args.seed + args.local_rank)

        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dataloader is not None:
            self.dataloader = dataloader
        else:
            self.dataloader = dataset.get_syntetic_loader(training_params["bs"], num_examples, data_shape, num_classes)
        self.model = self.model.to(self.device)

        # Define the optimizer
        if training_params is None:
            # Use standard params
            pass
        else:
            self.training_params = training_params
        self.train_epochs = training_params["train_epochs"]
        if optimizer is None:
            raise NotImplementedError

        self.model.optimizer = optimizer
        self.model.criterion = torch.nn.CrossEntropyLoss()

    def evaluate(self, starting_examles=100, max_examples=10000, increment=100, acc_thresh=0.9):
        shattering_dims = {}
        for num_examples in range(starting_examles, max_examples + increment, increment):
            print(f"[ESD] Training model using {num_examples} examples!")
            for _ in range(self.train_epochs):
                training_utils.train(self.model, self.dataloader, device=self.device)
            acc = training_utils.evaluate(self.model, self.dataloader, device=self.device)
            shattering_dims[num_examples] = acc

        shattering_dim = -1
        num_examples = sort(list(shattering_dims.keys()))
        for num_example in num_examples:
            acc = shattering_dims[num_example]
            if acc >= acc_thresh:
                shattering_dim = num_example
        return shattering_dim, shattering_dims
