import os
import re
import codecs
from setuptools import setup, find_packages

current_path = os.path.abspath(os.path.dirname(__file__))


def read_file(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return reader.read()


def get_requirements(*parts):
    with codecs.open(os.path.join(current_path, *parts), 'r', 'utf8') as reader:
        return list(map(lambda x: x.strip(), reader.readlines()))


def find_version(*file_paths):
    version_file = read_file(*file_paths)
    version_matched = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                version_file, re.M)
    if version_matched:
        return version_matched.group(1)
    raise RuntimeError('Unable to find version')


setup(
    name="see-rnn",
    version=find_version('see_rnn', '__init__.py'),
    packages=find_packages(exclude=['tests']),
    url="https://github.com/OverLordGoldDragon/see-rnn",
    license="MIT",
    author="OverLordGoldDragon",
    author_email="16495490+OverLordGoldDragon@users.noreply.github.com",
    description=('RNN weights, gradients, & activations visualization '
                 'in Keras & TensorFlow'),
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    keywords=(
        "rnn tensorflow keras visualization deep-learning lstm gru "
    ),
    install_requires=get_requirements('requirements.txt'),
    include_package_data=True,
    zip_safe=True,
    tests_require=["pytest>=4.0", "pytest-cov"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
