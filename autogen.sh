#!/bin/sh
# Run this to generate all the initial makefiles, etc.
# This was lifted from the Gimp, and adapted slightly by
# Raph Levien .

DIE=0

PROJECT="E-Cell"

(autoconf --version) < /dev/null > /dev/null 2>&1 || {
        echo
        echo "You must have autoconf installed to compile $PROJECT."
        DIE=1
}

(libtoolize --version) < /dev/null > /dev/null 2>&1 || {
        echo
        echo "You must have libtool installed to compile $PROJECT."
        DIE=1
}

(automake --version) < /dev/null > /dev/null 2>&1 || {
        echo
        echo "You must have automake installed to compile $PROJECT."
        DIE=1
}

if test "$DIE" -eq 1; then
        exit 1
fi

libtoolize -c --force

case $CC in
*xlc | *xlc\ * | *lcc | *lcc\ *) am_opt=--include-deps;;
esac

for dir in . libltdl dmtool ecell osogo bin modeleditor toollauncher
do 
  echo automake and autoconf for $dir
  (cd $dir; \
  aclocal; \
  automake --add-missing --gnu $am_opt; autoconf) 
done

for dir in . libltdl dmtool ecell osogo bin
do 
  echo generating headers for $dir
  (cd $dir; \
  autoheader -f)
done
