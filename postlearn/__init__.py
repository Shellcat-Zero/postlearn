# -*- coding: utf-8 -*-

from .reporter import *  # noqa

__author__ = 'Tom Augspurger'
__email__ = 'tom.augspurger88@gmail.com'
__version__ = '0.1.0'

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
