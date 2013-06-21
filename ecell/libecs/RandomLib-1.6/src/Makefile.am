noinst_LTLIBRARIES = libRandom.la
libRandom_la_SOURCES=Random.cpp

AM_CPPFLAGS=-I$(srcdir)/../include
AM_CXXFLAGS = -g -Wall -Wextra -O2 \
	-funroll-loops -finline-functions -fomit-frame-pointer

# if HAVE_SSE2
#  AM_CXXFLAGS+= -msse2
# endif
