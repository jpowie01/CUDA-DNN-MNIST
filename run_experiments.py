import os
import itertools
import subprocess

DYNAMIC = -1
SHARED_MEMORY_OFF = 0
SHARED_MEMORY_ON = 1


class Experiment:
    def __init__(self, batch_size, block_number, block_size, shared):
        self.batch_size = batch_size
        self.block_number = block_number
        self.block_size = block_size
        self.shared = shared

    def __repr__(self):
        return '<Experiment - BatchSize: {} BlockNumber: {} BlockSize: {} SharedMemory: {}>'.format(
            self.batch_size,
            self.block_number,
            self.block_size,
            self.shared,
        )


EXPERIMENTS = [
    # Check what will happen when we will change Batch Size on classic implementation
    Experiment(batch_size=1, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=2, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=4, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=8, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=16, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=32, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=64, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=1024, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),

    # Check what will happen when we will change Block Size on classic implementation and fixed batch
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=1, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=2, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=4, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=8, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=64, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=128, shared=SHARED_MEMORY_OFF),

    # Check what will happen when we will change Number of Blocks on classic implementation and fixed batch
    Experiment(batch_size=128, block_number=1, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=2, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=4, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=8, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=16, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=32, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=64, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=128, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=256, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=512, block_size=16, shared=SHARED_MEMORY_OFF),

    # Check what will happen when we will change Batch Size on implementation with Shared Memory
    Experiment(batch_size=1, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=2, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=4, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=8, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=16, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=32, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=64, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=256, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=512, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=1024, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_ON),
]

for experiment in EXPERIMENTS:
    print('Running {}...'.format(experiment))
    os.environ['BATCH_SIZE'] = str(experiment.batch_size)
    os.environ['TENSOR2D_MULTIPLY_BLOCK_NUMBER'] = str(experiment.block_number)
    os.environ['TENSOR2D_MULTIPLY_BLOCK_SIZE'] = str(experiment.block_size)
    os.environ['TENSOR2D_MULTIPLY_SHARED_MEMORY'] = str(experiment.shared)
    filename = 'logs/BATCH_{}__B_NUMBER_{}__B_SIZE_{}__SHARED_{}'.format(
        experiment.batch_size, experiment.block_number,
        experiment.block_size, experiment.shared
    )
    os.environ['LOG_FILE_NAME'] = '{}.csv'.format(filename)
    with open('{}.log'.format(filename), 'w') as execution_dump:
        subprocess.call("make run".split(), stdout=execution_dump)

