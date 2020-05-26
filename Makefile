CXXFLAGS += -std=c++11 -O2

# Default to using system's default version of python
PYTHON_BIN     ?= python3
PYTHON_CONFIG  := $(PYTHON_BIN)-config
PYTHON_INCLUDE ?= $(shell $(PYTHON_CONFIG) --includes)
CXXFLAGS       += $(PYTHON_INCLUDE)
LDFLAGS        += $(shell $(PYTHON_CONFIG) --libs)

EIGEN_INCLUDE = /usr/include/eigen3
CXXFLAGS += -I$(EIGEN_INCLUDE)

# Either finds numpy or set -DWITHOUT_NUMPY
CXXFLAGS        += $(shell $(PYTHON_BIN) $(CURDIR)/src/numpy_flags.py)
WITHOUT_NUMPY   := $(findstring $(CXXFLAGS), WITHOUT_NUMPY)

all: bin/one bin/two bin/three bin/four

bin/one: src/one.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

bin/two: src/two.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

bin/three: src/three.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

bin/four: src/four.cc
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f bin/*