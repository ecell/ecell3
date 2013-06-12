sh autogen.sh;CPPFLAGS="-I$1/include" CFLAGS="-L$1/lib" ./configure --prefix=$1;make;make install
