import setuptools

# Load the long_description from README.md
with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Qpykan",
    version="0.0.5",
    author="Modifications made by William Troy. Original package made by Ziming Liu.",
    author_email="troywilliame@gmail.com",
    description="Modified version of the original pykan Kolmogorov Arnold Networks. Modifications were made for speed testing of the Bezier KAN. The original repo for pykan can be found at https://github.com/KindXiaoming/pykan.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
