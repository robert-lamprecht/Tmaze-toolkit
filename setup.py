from setuptools import setup, find_packages

setup(
    name="TmazeAnalysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "matplotlib",
        "scipy",
        "pandas",
        "tqdm",
    ],
    author="Robert Lamprecht",
    author_email="rlampre@emory.edu",
    description="A package for analyzing Tmaze data from experimental videos",
    keywords="video analysis, motion detection, behavioral experiments",
    python_requires=">=3.7",
)
