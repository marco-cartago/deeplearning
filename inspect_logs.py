import argparse
import pickle

import matplotlib.pyplot as plt

from mlog import ModelLog

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training log of Mamba.")
    parser.add_argument("-f", "--file", type=str, help="Path to logging file.")
    parser.add_argument("-p", "--plot", type=str, 
                        help="Plot history of training. Requires subinterval specifications using commas `from,to`.")
    return parser.parse_args()

def load_log(model_log_path: str) -> ModelLog:
    with open(model_log_path, 'rb') as logfile:
        log: ModelLog = pickle.load(logfile)
        return log

def plot_loss_history(ml: ModelLog, start=0, stop: int | None = None, crop=40.):
    iters = list(range(1, len(ml.e_train_loss)+1))
    plt.plot(iters[start:stop], ml.e_train_loss[start:stop])
    plt.ylim(0, crop)
    plt.title('Training Loss History')
    plt.ylabel('Cross-Entropy Loss')
    plt.xlabel('Iterations')
    plt.savefig('figures/training_loss.png')

if __name__ == '__main__':
    args = parse_arguments()
    logfile = args.file
    plot_lims = list(map(int, (args.plot.split(','))))
    modellog = load_log(logfile)
    plot_loss_history(modellog, start=plot_lims[0], stop=plot_lims[1])