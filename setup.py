from setuptools import setup, Extension
import pybind11

# Find the pybind11 include directory
pybind11_include = pybind11.get_include()

# Define the extension module
bezier_sse_eigen_module = Extension(
    'quantum_kan',
    sources=['cpp/quantum_kan.cpp'],
    include_dirs=[
        pybind11_include,
        '/usr/include/eigen3'
    ],
    library_dirs=['/usr/local/lib', '/usr/lib'],  # Add your SymEngine library path if different
    libraries=['symengine', 'gmp', 'pthread', 'openblas'],
    extra_compile_args=['-O3', '-Wall', '-std=c++17', '-fPIC']
)

# Setup script
setup(
    name='quantum_kan',
    version='0.1.0',
    author='William Troy',
    author_email='troywilliame@gmail.com',
    description='A package for the current implimentation of Quantum KAN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_modules=[bezier_sse_eigen_module],
    install_requires=['pybind11'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    zip_safe=False,
)

# c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` bezier_sse_eigen_aligned.cpp -o bezier_sse_eigen_aligned`python3-config --extension-suffix` -I /usr/include/eigen3 -lsymengine -lgmp -lpthread -lopenblas