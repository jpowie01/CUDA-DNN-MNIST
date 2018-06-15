import os
import argparse
import itertools
import subprocess

from experiments import EXPERIMENTS, get_experiment_name


parser = argparse.ArgumentParser(description='Run experiments on GPU.')
parser.add_argument('--logs-dir', type=str, default='logs', help='Directory for logs.')
args = parser.parse_args()
logs_dir = args.logs_dir

for experiment in EXPERIMENTS:
    print('Running {}...'.format(experiment))
    os.environ['NUMBER_OF_EPOCHS'] = '2'  # First epoch may be slower as it needs to allocate memory dynamically
    os.environ['BATCH_SIZE'] = str(experiment.batch_size)
    os.environ['TENSOR2D_MULTIPLY_BLOCK_NUMBER'] = str(experiment.block_number)
    os.environ['TENSOR2D_MULTIPLY_BLOCK_SIZE'] = str(experiment.block_size)
    os.environ['TENSOR2D_MULTIPLY_SHARED_MEMORY'] = str(experiment.shared)
    os.makedirs(logs_dir, exist_ok=True)
    filename = os.path.join(logs_dir, get_experiment_name(experiment))
    os.environ['LOG_FILE_NAME'] = '{}.csv'.format(filename)
    with open('{}.log'.format(filename), 'w') as execution_dump:
        subprocess.call('make run'.split(), stdout=execution_dump)

