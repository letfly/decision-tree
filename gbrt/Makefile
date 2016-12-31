export CC = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2

# specify tensor path
BIN = 
OBJ = gboost.o
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm

gboost.o: booster/*.cc

$(BIN) :
	   $(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cc %.o %.c, $^)
$(OBJ) :
	   $(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cc %.c, $^) )
install:
	   cp -f -r $(BIN) $(INSTALL_PATH)
clean:
	   $(RM) $(OBJ) $(BIN) *~
