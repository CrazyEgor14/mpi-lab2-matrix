CXX = mpic++
CXXFLAGS = -std=c++11 -O2
TARGET = lab2_matrix
SOURCES = lab2_matrix.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

run: $(TARGET)
	mpiexec -n 5 ./$(TARGET)

test-sequential:
	g++ -std=c++11 -o test_seq test_sequential.cpp && ./test_seq

clean:
	rm -f $(TARGET) test_seq

.PHONY: run test-sequential clean
