CC = gcc
CFLAGS = -Wall -O3

mlc: main.c
	$(CC) $(CFLAGS) main.c -o mlc

clean:
	rm -f mlc *.o