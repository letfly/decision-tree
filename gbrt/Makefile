export CC = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2

# specify tensor path
BIN = gboost
OBJ = gboost.o
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm

gboost.o: booster/*.cc
gboost: regression/gboost_reg_main.cc gboost.o

$(BIN) :
	   $(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cc %.o %.c, $^) -I ./
$(OBJ) :
	   $(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cc %.c, $^) ) -I ./
install:
	   cp -f -r $(BIN) $(INSTALL_PATH)
clean:
	   $(RM) $(OBJ) $(BIN) *~
