from setuptools import setup, find_packages


def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires


requires = open_requirements('requirements.txt')
setup(
    name='pmt_analysis',
    version='0.0.1',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='Comprehensive analysis for PMT characterization data.',
    long_description=open('README.md').read(),
    install_requires=requires,
    url='https://github.com/a1exndr/PMT_Analysis',
    author='Alexander Bismark',
    author_email='alexander.bismark@alumni.cern'
)
