from setuptools import setup, find_packages

setup(
    name             = "sangchu",
    version          = "0.1.0",
    description      = "Green leaf area measurement for greenhouse pot crops",
    author           = "agronomy4future",
    url              = "https://github.com/agronomy4future/sangchu",
    packages         = find_packages(),
    python_requires  = ">=3.8",
    install_requires = [
        "opencv-python",
        "numpy",
        "pandas",
    ],
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
