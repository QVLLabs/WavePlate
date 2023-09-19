from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

__version__ = "0.0.5"
DESCRIPTION = 'A python library for simulating optics'
LONG_DESCRIPTION = 'This package allow you to simulate photons behavior and maniplute them with optical elements'

# Setting up
setup(
    name="waveplate",
    version=__version__,
    author="Quantum Visionary Labs",
    author_email="<qvllabs@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    license='MIT', 
    install_requires=['numpy'],
    keywords=['python', 'optics', 'photonics', 'photons', 'physics', 'quantum mechanics'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
