#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

requirements = [
    'pandas',
    'scikit-learn'
]

test_requirements = [
    'pytest'
]

setup(
    name='postlearn',
    version='0.1.0',
    description="After fitting",
    long_description=readme,
    author="Tom Augspurger",
    author_email='tom.augspurger88@gmail.com',
    url='https://github.com/TomAugspurger/postlearn',
    packages=[
        'postlearn',
    ],
    package_dir={'postlearn':
                 'postlearn'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='postlearn',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
