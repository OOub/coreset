import setuptools
import runpy
import os

root = os.path.dirname(os.path.realpath(__file__))
version = runpy.run_path(os.path.join(root, "coreset", "version.py"))["version"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coresets",
    version=version,
    author="Omar Oubari",
    author_email="omaroubari@gmail.com",
    description="Data summarization via lightweight coresets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OOub/coresets",
    include_package_data=False,
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
