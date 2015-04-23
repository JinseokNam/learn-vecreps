CC := g++ -std=c++11
CC_OPTS := -O3
CFLAGS := -c -Wall
INC := -I/usr/local/include -I/usr/include -I${HOME}/local/include -I./include
LIB := -L/usr/local/lib -L/usr/lib -L${HOME}/local/lib
LDFLAGS := -lboost_thread -lboost_system -lglog -lopenblas -lpthread
SRCDIR := src
BUILDDIR := bin
EX_W2V_SRCDIR := examples/word2vec
EX_P2V_SRCDIR := examples/par2vec
SRCS:=$(wildcard $(SRCDIR)/*.cpp)
EX_W2V_SRCS:=$(wildcard $(EX_W2V_SRCDIR)/*.cpp)
EX_P2V_SRCS:=$(wildcard $(EX_P2V_SRCDIR)/*.cpp)
HEADERS:=$(wildcard include/*.hpp)
OBJS:=$(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(SRCS))
EX_W2V_OBJS:=$(patsubst $(EX_W2V_SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(EX_W2V_SRCS))
EX_P2V_OBJS:=$(patsubst $(EX_P2V_SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(EX_P2V_SRCS))
W2V_TARGET := $(BUILDDIR)/test_word2vec
P2V_TARGET := $(BUILDDIR)/test_par2vec

.DEFAULT: all

all: $(SRCS) $(W2V_TARGET) $(P2V_TARGET)

$(P2V_TARGET): $(OBJS) $(EX_P2V_OBJS)
	${CC} $^ -o $(P2V_TARGET) $(CC_OPTS) $(INC) $(LIB) $(LDFLAGS)

$(W2V_TARGET): $(OBJS) $(EX_W2V_OBJS)
	${CC} $^ -o $(W2V_TARGET) $(CC_OPTS) $(INC) $(LIB) $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -c -o $@ $(INC)

$(BUILDDIR)/%.o: $(EX_W2V_SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -c -o $@ $(INC)

$(BUILDDIR)/%.o: $(EX_P2V_SRCDIR)/%.cpp $(HEADERS)
	$(CC) $< $(CC_OPTS) $(CFLAGS) -c -o $@ $(INC)

.PHONY: clean
clean:
	rm $(BUILDDIR)/*.o $(P2V_TARGET) $(W2V_TARGET)
