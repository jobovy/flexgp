from setuptools import setup #, Extension
import os, os.path
import re

longDescription= ""


setup(name='flexgp',
      version='1.',
      description='Gaussian Process code with an emphasis on flexibility rather than speed',
      author='Jo Bovy',
      author_email='bovy@ias.edu',
      license='New BSD',
      long_description=longDescription,
      url='https://github.com/jobovy/flexgp',
      package_dir = {'flexgp/': ''},
      packages=['flexgp'],
      dependency_links = ['https://github.com/jobovy/bovy_mcmc/tarball/master#egg=bovy_mcmc'],
      install_requires=['bovy_mcmc']
      )
