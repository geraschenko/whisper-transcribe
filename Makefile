# Makefile for whisper transcribe binary
# Build system for transcribe.cpp using existing whisper.cpp build

# Paths
WHISPER_CPP_DIR = /home/anton/git/whisper.cpp
WHISPER_BUILD_DIR = $(WHISPER_CPP_DIR)/build
EXAMPLES_DIR = $(WHISPER_CPP_DIR)/examples
BUILD_DIR = build

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread -Wall -Wextra
SDL2_CFLAGS = $(shell pkg-config --cflags sdl2)
INCLUDES = -I$(WHISPER_CPP_DIR)/include \
           -I$(WHISPER_CPP_DIR) \
           -I$(EXAMPLES_DIR) \
           -I$(WHISPER_CPP_DIR)/ggml/include \
           $(SDL2_CFLAGS)

# Library paths and linking
LIBDIRS = -L$(WHISPER_BUILD_DIR)/src \
          -L$(WHISPER_BUILD_DIR)/ggml/src

# Core libraries needed for transcription
LIBS = -lwhisper -lggml -lggml-base

# Add CUDA library if available (check if built with CUDA)
ifneq ($(wildcard $(WHISPER_BUILD_DIR)/ggml/src/ggml-cuda),)
    LIBDIRS += -L$(WHISPER_BUILD_DIR)/ggml/src/ggml-cuda
    LIBS += -lggml-cuda
endif

# SDL2 for audio capture
SDL2_LIBS = $(shell pkg-config --libs sdl2)


# Target and source
TARGET = $(BUILD_DIR)/transcribe
SOURCE = transcribe.cpp

# Common source files from whisper.cpp examples
COMMON_SOURCES = $(EXAMPLES_DIR)/common.cpp \
                 $(EXAMPLES_DIR)/common-ggml.cpp \
                 $(EXAMPLES_DIR)/common-whisper.cpp \
                 $(EXAMPLES_DIR)/common-sdl.cpp

# Common object files
COMMON_OBJS = $(BUILD_DIR)/common.o $(BUILD_DIR)/common-ggml.o $(BUILD_DIR)/common-whisper.o $(BUILD_DIR)/common-sdl.o

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Common object files compilation
$(BUILD_DIR)/common.o: $(EXAMPLES_DIR)/common.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common-ggml.o: $(EXAMPLES_DIR)/common-ggml.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common-whisper.o: $(EXAMPLES_DIR)/common-whisper.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/common-sdl.o: $(EXAMPLES_DIR)/common-sdl.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Main target
$(TARGET): $(SOURCE) $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) $(COMMON_OBJS) $(LIBDIRS) $(LIBS) $(SDL2_LIBS) -o $(TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the transcribe binary (default)"
	@echo "  clean    - Remove built files"
	@echo "  help     - Show this help message"

# Phony targets
.PHONY: all clean install help