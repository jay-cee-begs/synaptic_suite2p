from setuptools import setup, find_packages

setup(
    name='synaptic_suite2p', 
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
        # List your project dependencies here
    
)

"""File can be run in terminal with 
python setup.py develop"""