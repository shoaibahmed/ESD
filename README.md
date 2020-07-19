# ESD

A simple PyTorch library for computing empirical shattering dimension of a given network. 

The library naturally supports distributed models which includes both DataParallel and DistributedDataParallel. The user just has to initialize the environment and pass in the distributed model to the library.
The library also supports using synthetic data instead of actual data samples. In case an actual dataset is provided, the library just replaces the targets with randomly generated targets.

## Installation

To install the library, use the following command:

```bash
python setup.py install
```

## Examples

An example script is provided in the examples directory. The example iterate over different pretrained PyTorch models and uses them to compute the ESD.
The example also provides distributed example along with usage of synthetic dataset.

## TODOs

- Test and verify correctness of distributed training

## License:

MIT

## Issues/Feedback:

In case of any issues, feel free to drop me an email or open an issue on the repository.

Email: **shoaib_ahmed.siddiqui@dfki.de**
