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
CXXFLAGS        += $(shell $(PYTHON_BIN) $(CURDIR)/numpy_flags.py)
WITHOUT_NUMPY   := $(findstring $(CXXFLAGS), WITHOUT_NUMPY)

all: one two three

one: one.cc
	$(CXX) -o one one.cc $(CXXFLAGS) $(LDFLAGS)

two: two.cc
	$(CXX) -o two two.cc $(CXXFLAGS) $(LDFLAGS)

three: three.cc
	$(CXX) -o three three.cc $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f one one.png two three