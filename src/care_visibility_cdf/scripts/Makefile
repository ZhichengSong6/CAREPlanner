# measured_arm_numeric.py loads this library from its own directory.
CXXFLAGS ?= -O3
EIGEN_CFLAGS ?= $(shell pkg-config --cflags eigen3)

measured_arm_geometry.so: measured_arm_geometry.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EIGEN_CFLAGS) -std=c++17 -fPIC -shared $(LDFLAGS) -o $@ $< $(LDLIBS)
