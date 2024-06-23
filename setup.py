from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    README = fh.read()

setup(
    name="Overcomplete",
    version="0.0.1",
    description="Personal toolbox for experimenting with Overcomplete Dictionnary Learning (or SAE)",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas Fel",
    author_email="thomas_fel@brown.edu",
    license="MIT",
    install_requires=['numpy', 'matplotlib', 'torch', 'torchvision', 'timm'],
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
