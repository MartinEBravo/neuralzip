CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -shared -fPIC

COMPRESSION_DIR := compression_algorithms
NATIVE_TARGETS := \
	$(COMPRESSION_DIR)/arithmetic_coder.so \
	$(COMPRESSION_DIR)/huffman_coder.so

.PHONY: native clean-native

native: $(NATIVE_TARGETS)

$(COMPRESSION_DIR)/arithmetic_coder.so: $(COMPRESSION_DIR)/arithmetic_coder.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

$(COMPRESSION_DIR)/huffman_coder.so: $(COMPRESSION_DIR)/huffman_coder.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean-native:
	rm -f $(NATIVE_TARGETS)
