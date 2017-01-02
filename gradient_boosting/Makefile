export CC = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2

# specify tensor path
BIN = main
OBJ = 
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm

main: regression/gboost_reg_main.cc

$(BIN) :
	   $(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cc %.o %.c, $^) -I ./
$(OBJ) :
	   $(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cc %.c, $^) ) -I ./
install:
	   cp -f -r $(BIN) $(INSTALL_PATH)
clean:
	   $(RM) $(OBJ) $(BIN) *~
