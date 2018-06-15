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


# Check what will happen when we will change Batch Size on classic implementation
BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF__PART1 = [
    Experiment(batch_size=1, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=2, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=4, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=8, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=16, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=32, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
]
BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF__PART2 = [
    Experiment(batch_size=64, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=1024, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
]
BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF = BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF__PART1 + \
                                            BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF__PART2

# Check what will happen when we will change Block Size on classic implementation and fixed batch
BLOCKS_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF = [
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=1, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=2, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=4, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=8, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=DYNAMIC, block_size=32, shared=SHARED_MEMORY_OFF),
]

# Check what will happen when we will change Number of Blocks on classic implementation and fixed batch
BLOCK_NUMBER_EXPERIMENTS__SHARED_MEMORY_OFF = [
    Experiment(batch_size=128, block_number=4, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=8, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=16, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=32, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=64, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=128, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=256, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=512, block_size=16, shared=SHARED_MEMORY_OFF),
]

# Check what will happen when we will change Batch Size on implementation with Shared Memory
BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_ON = [
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

# Combine several options together
BATCH_SIZE_BLOCK_NUMBER_AND_BLOCK_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF = [
    Experiment(batch_size=128, block_number=16, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=32, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=16, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=128, block_number=32, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=16, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=32, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=16, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=256, block_number=32, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=16, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=32, block_size=16, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=16, block_size=32, shared=SHARED_MEMORY_OFF),
    Experiment(batch_size=512, block_number=32, block_size=32, shared=SHARED_MEMORY_OFF),
]

# Combine several options together and optimization based on shared memory
BATCH_SIZE_BLOCK_NUMBER_AND_BLOCK_SIZE_EXPERIMENTS__SHARED_MEMORY_ON = [
    Experiment(batch_size=128, block_number=32, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=128, block_number=32, block_size=32, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=256, block_number=32, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=256, block_number=32, block_size=32, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=512, block_number=32, block_size=16, shared=SHARED_MEMORY_ON),
    Experiment(batch_size=512, block_number=32, block_size=32, shared=SHARED_MEMORY_ON),
]

EXPERIMENTS = BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF + BLOCKS_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF + \
              BLOCK_NUMBER_EXPERIMENTS__SHARED_MEMORY_OFF + BATCH_SIZE_EXPERIMENTS__SHARED_MEMORY_ON + \
              BATCH_SIZE_BLOCK_NUMBER_AND_BLOCK_SIZE_EXPERIMENTS__SHARED_MEMORY_OFF + \
              BATCH_SIZE_BLOCK_NUMBER_AND_BLOCK_SIZE_EXPERIMENTS__SHARED_MEMORY_ON


def get_experiment_name(experiment):
    return 'BATCH_{}__B_NUMBER_{}__B_SIZE_{}__SHARED_{}'.format(
        experiment.batch_size, experiment.block_number,
        experiment.block_size, experiment.shared
    )
