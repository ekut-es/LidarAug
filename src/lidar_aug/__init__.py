import os
import platform
import multiprocessing

num_threads = multiprocessing.cpu_count(
) if 'OMP_NUM_THREADS' not in os.environ else os.environ['OMP_NUM_THREADS']
if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import torch

from . import point_cloud
from . import transformations
from . import weather_simulations
from . import evaluation
from .python import utils
from .python import visualization
from .python import augmentations

if platform.system() == 'Darwin':
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
