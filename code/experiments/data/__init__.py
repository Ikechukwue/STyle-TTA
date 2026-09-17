"""
xAILab Bamberg
University of Bamberg

@description:
Data module initialization.

@author: Sebastian Doerrich
"""

from code.experiments.data._factory import create_dataset, CustomDataset
from code.experiments.data.imagefolder import ImageFolder
from code.experiments.data._constants import NORMALIZATION_MEAN, NORMALIZATION_STD, NUM_CLASSES, TASK_TYPE, DATASET_SPLITS
from code.experiments.data.cifar import CIFAR
from code.experiments.data.imagenet_variants import ImageNet
from code.experiments.data.pacs import PACS
from code.experiments.data.vlcs import VLCS
from code.experiments.data.domainnet import DomainNet
from code.experiments.data.office_home import OfficeHome
from code.experiments.data.office_caltech import OfficeCaltech
from code.experiments.data.visda17 import VisDA17
from code.experiments.data.visual_decathlon import VisualDecathlon
from code.experiments.data.terra_incognita import TerraIncognita
from code.experiments.data.colored_mnist import ColoredMNIST
from code.experiments.data.nico import NICO
from code.experiments.data.metashift import MetaShift
from code.experiments.data.openmibood import OpenMIBOOD

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
