from setuptools import find_packages, setup

setup(
    name="recommend",
    author="Marek Kadlcik, Petra Kratka, Andrej Kubanda, Adam Hajek",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": ["recommend=recommend.main:main"],
    },
)
