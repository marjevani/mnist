import argparse
from backend import Inference_manager, mnist_object

parser = argparse.ArgumentParser(description='Guess your digit using convectional neural network')

parser.add_argument('--train', type=bool, default=False, help='use for training your CNN. drink some coffee meanwhile')
args = parser.parse_args()


if args.train:
    mnist_object.train_mnist_CNN()
else:
    Inference_manager.Inference_manager()