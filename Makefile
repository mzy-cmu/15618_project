# Compiler
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Target executable
TARGET = ParaFaultSim

# Source files
SRCS = ParaFaultSim.cpp Circuit.cpp Evaluate.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program with a dynamic benchmark number
run: $(TARGET)
	@if [ -z "$(bench_num)" ]; then \
		echo "Error: bench_num is not set. Use 'make run bench_num=<value>'"; \
		exit 1; \
	fi
	./$(TARGET) -f Benchmarks/$(bench_num).bench -t Tests/$(bench_num).test > Results/$(bench_num).result
