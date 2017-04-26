import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "lspiv_toolkit",
    version = "0.0.1",
    author = "Christopher 'ckt' Tomaszewski",
    author_email = "christomaszewski@gmail.com",
    description = ("A library for writing LSPIV programs"),
    license = "BSD",
    keywords = "lspiv piv",
    url = "https://github.com/christomaszewski/lspiv_toolkit.git",
    packages=['lspiv_toolkit', 'tests', 'examples'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)