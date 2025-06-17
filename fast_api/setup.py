from setuptools import setup, find_packages


with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="api",
    version="0.1.0",
    install_requires=required,
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
)
