# -*- coding: utf-8 -*-

from .reporter import *  # noqa
from . import utils  # noqa

__author__ = 'Tom Augspurger'
__email__ = 'tom.augspurger88@gmail.com'
__version__ = '0.1.0'

from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions
