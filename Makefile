CC := g++ -std=c++11
CC_OPTS := -O3
CFLAGS := -c -Wall
INC := -I/usr/local/include -I./include
LIB := -L/usr/local/lib
LDFLAGS := -lboost_thread -lboost_system -lglog -lopenblas
SRCDIR := src
BUILDDIR := bin
SRCS:=$(wildcard $(SRCDIR)/*.cpp)
HEADERS:=$(wildcard include/*.hpp)
OBJS:=$(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRCS))
TARGET := $(BUILDDIR)/word2vec_test

.DEFAULT: all

all: $(SRCS) $(TARGET)

$(TARGET): $(OBJS)
	${CC} $^ -o $(TARGET) $(CC_OPTS) $(INC) $(LIB) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -c -o $@ $(INC)

.PHONY: clean
clean:
	rm $(BUILDDIR)/*.o $(TARGET)
