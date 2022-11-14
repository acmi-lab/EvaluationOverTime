from setuptools import setup, find_packages


setup(
    name='emdot',
    version='0.1.0',
    description='Evaluation of Medical Datasets Over Time package.',
    url='https://github.com/acmilab/EvaluationOverTime',
    packages=find_packages(),
    # packages=[
    #     'emdot',
    # ],
    install_requires=[
        'matplotlib==3.2.0',
        'numpy>=1.21.5',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.4',
        'tqdm>=4.64.0',
        'jupyter',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ]
)