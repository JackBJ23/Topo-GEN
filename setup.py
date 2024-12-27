from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="topogen",
    version="0.1.0",
    author="Jack B. Jedlicki",
    author_email="jackbj@berkeley.edu",
    description="A package for topological regularizers and utilities",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JackBJ23/Topo-GEN",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=install_requires,
)
