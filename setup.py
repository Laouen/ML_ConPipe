from setuptools import setup

setup(
    name='ConPipe',
    version='1.0',
    description='A framework to create and run machine learning processing flows and experiment from YAML graphs specifications',
    author='Laouen Mayal Louan belloli',
    author_email='laouen.belloli@gmail.com',
    package=['ConPipe'],
    scripts=['scripts/run_ml_experiment.py'],
    install_requires=[
        'PyYAML',
        'pandas',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'graph-theory'
    ]
)
