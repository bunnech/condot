#!/usr/bin/python3

# imports
from setuptools import setup, find_packages

setup(name='condot',
      version='1.0',
      description='Supervised Training of Conditional Monge Maps',
      url='https://github.com/bunnech/condot',
      author='Charlotte Bunne',
      author_email='bunnec@ethz.ch',
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[],
      zip_safe=False)
