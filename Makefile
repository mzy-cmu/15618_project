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

bench_num = s208
# Run the program
run: $(TARGET)
	./$(TARGET) -f Benchmarks/$(bench_num).bench > Results/$(bench_num).result
