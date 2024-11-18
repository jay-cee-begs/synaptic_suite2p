from setuptools import setup, find_packages
import chardet

# Detect file encoding
def detect_encoding(filename):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def parse_requirements(filename, encoding):
    with open(filename, 'r', encoding=encoding) as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


encoding = detect_encoding('requirements.txt')  # Or 'requirements.yaml'
print(f"Detected encoding: {encoding}")
requirements = parse_requirements('requirements.txt', encoding)

setup(
    name='synaptic_suite2p', 
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=requirements
        # List your project dependencies here
    
)

"""File can be run in terminal with 
python setup.py develop"""