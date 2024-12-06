# Compiler
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
NVCCFLAGS = -std=c++17 -O2

# Target executable
TARGET = ParaFaultSim

# Source files
CPP_SRCS = ParaFaultSim.cpp Circuit.cpp Evaluate.cpp
CUDA_SRCS = ParaFaultSim.cu

# Object files
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# All object files
OBJS = $(CPP_OBJS) $(CUDA_OBJS)

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Rule to build C++ object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to build CUDA object files
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(CPP_OBJS) $(CUDA_OBJS) $(TARGET)

# Run the program with a dynamic benchmark number
run: $(TARGET)
	@if [ -z "$(bench_num)" ]; then \
		echo "Error: bench_num is not set. Use 'make run bench_num=<value>'"; \
		exit 1; \
	fi
	./$(TARGET) -f Benchmarks/$(bench_num).bench -t Tests/$(bench_num).test > Results/$(bench_num).result
