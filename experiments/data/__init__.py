"""
xAILab Bamberg
University of Bamberg

@description:
Data module initialization.

@author: Sebastian Doerrich
"""

from experiments.data._factory import create_dataset, CustomDataset
from experiments.data.imagefolder import ImageFolder
from experiments.data._constants import NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_CLASSES, TASK_TYPE, DATASET_SPLITS
from experiments.data.cifar import CIFAR
from experiments.data.imagenet_variants import ImageNet
from experiments.data.pacs import PACS
from experiments.data.vlcs import VLCS
from experiments.data.domainnet import DomainNet
from experiments.data.office_home import OfficeHome
from experiments.data.office_caltech import OfficeCaltech
from experiments.data.visda17 import VisDA17
from experiments.data.visual_decathlon import VisualDecathlon
from experiments.data.terra_incognita import TerraIncognita
from experiments.data.colored_mnist import ColoredMNIST
from experiments.data.nico import NICO
from experiments.data.metashift import MetaShift
from experiments.data.openmibood import OpenMIBOOD

__all__ = [
    'create_dataset',
    'CustomDataset',
    'ImageFolder',
    'NORMALIZATION_MEAN',
    'NORMALIZATION_STD',
    'NUM_CLASSES',
    'TASK_TYPE',
    'DATASET_SPLITS',
    'CIFAR',
    'ImageNet',
    'PACS',
    'VLCS',
    'DomainNet',
    'OfficeHome',
    'OfficeCaltech',
    'VisDA17',
    'VisualDecathlon',
    'TerraIncognita',
    'ColoredMNIST',
    'NICO',
    'MetaShift',
    'OpenMIBOOD',
]
