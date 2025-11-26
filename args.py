"""## Arguments"""
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='poison conformal predictions')

    parser.add_argument('-device', default='gpu', type=str, help='cuda/cpu')
    parser.add_argument('-moddir', type=str, default="./models", help="model directory")
    parser.add_argument('-outdir', default='./outputs', type=str, help='output directory')
    parser.add_argument('-logdir', default='./logs', type=str, help='log directory')
    parser.add_argument('-imgdir', default='./images', type=str, help='images directory')
    parser.add_argument('-resdir', default='./results', type=str, help='results directory')
    parser.add_argument('-initdir', default='./inits', type=str, help='inits directory')

    # training
    parser.add_argument('-num_classes', default=2, type=int, help='number of class')
    parser.add_argument('-model', default='autoencoder', type=str, help='model architecture')
    parser.add_argument('-lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('-optimizer', default='sgd', type=str, help='initial optimizer')
    parser.add_argument('-dataset', default='simulated', type=str, help='craft dataset')
    parser.add_argument('-batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-epochs', default=3, type=int, help='number of pre-training epochs')
    parser.add_argument('-train_size', default=1.0, type=float, help='training set size')
    parser.add_argument('-num_workers', default=2, type=int, help='number of workers')
    parser.add_argument('-loss', default='ce', type=str, help='loss function')
    parser.add_argument('-log_interval', default=8, type=int, help='log file interval')

    # vba parameters

    return parser.parse_args()

class Args:
    def __init__(self, device, moddir, outdir, logdir, imgdir, resdir, initdir, num_classes, model, lr, optimizer, dataset, batch_size, epochs, train_size, num_workers, loss,log_interval):
        self.device = device
        self.moddir = moddir
        self.outdir = outdir
        self.logdir = logdir
        self.imgdir = imgdir
        self.resdir = resdir
        self.initdir = initdir
        self.num_classes = num_classes
        self.model = model
        self.lr = lr
        self.optimizer = optimizer
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_size = train_size
        self.num_workers = num_workers
        self.loss = loss
        self.log_interval = log_interval