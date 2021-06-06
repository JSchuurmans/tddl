from setuptools import setup, find_packages

base_requirements = [
    'torch',
    'torchvision',
    'tensorly',
    'tensorly-torch',
]

dev_requirements = [
    'jupyterlab',
    'kaggle',
    # 'pytest',
    # 'pytest-flake8',
    # 'pytest-cov',
]

setup(
    name='tddl',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=base_requirements,
    extras_require={
        'dev': dev_requirements,
    },
)
