# Compile
 c++ -O3 -Wall -shared -std=c++17 -fPIC `python3 -m pybind11 --includes` quant
um_kan.cpp -o quantum_kan`python3-config --extension-suffix` -I /usr/include/eigen3 -lsymengine -lgmp -lpthread -lopenblas

# Compile command for pg testing
c++ -O3 -pg -Wall -std=c++17  quantum_kan.cpp -o quantum_kan  -I /usr/include/eigen3 -lsymengine -lgmp -lpthread -lopenblas

# Profiling commands
gprof ./quantum_kan > output.txt

gprof quantum_kan gmon.out | gprof2dot -w | dot -Gdpi=250 -Tpng -o output.png
