from setuptools import setup

setup(
    name='pat',
    version='0.0.1',    
    description='Experiment',
    url='https://github.com/ndronen/PATExperiment',
    author='Nicholas Dronen, Helian Feng',
    author_email='ndronen@gmail.com',
    license='BSD 2-clause',
    packages=['pat'],
    install_requires=[
        'torch', 'numpy', 'scikit-dimension', 'scikit-learn', 'plotly'
    ],
    classifiers=[]
)
