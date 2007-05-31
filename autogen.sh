#!/bin/sh
# Run this to generate all the initial makefiles, etc.
# This was lifted from the Gimp, and adapted slightly by
# Raph Levien .

DIE=0

PROJECT="E-Cell"

if test -z "$AUTOCONF"; then
    if autoconf --version < /dev/null > /dev/null 2>&1; then
  echo
  echo "You must have autoconf installed to compile $PROJECT."
  DIE=1
}

if libtoolize --version < /dev/null > /dev/null 2>&1; then
  LIBTOOLIZE=libtoolize
elif  glibtoolize --version < /dev/null > /dev/null 2>&1; then
  LIBTOOLIZE=glibtoolize
else
  echo
  echo "You must have libtool installed to compile $PROJECT."
  DIE=1
fi

(automake --version) < /dev/null > /dev/null 2>&1 || {
  echo
  echo "You must have automake installed to compile $PROJECT."
  DIE=1
}

if test "$DIE" -eq 1; then
  exit 1
fi

case $CC in
*xlc | *xlc\ * | *lcc | *lcc\ *) am_opt=--include-deps;;
esac

for dir in . libltdl dmtool ecell
  do 
  echo -n "Running autotools for $dir ... "
  (cd $dir; \
  { echo -n 'libtoolize '; $LIBTOOLIZE -c --force --automake; } && \
  { echo -n 'aclocal '; aclocal; } && \
  { echo -n 'autoheader '; autoheader -f ; } && \
  { echo -n 'automake ';  automake --add-missing --gnu $am_opt; } && \
  { echo -n 'autoconf '; autoconf; } && \
  echo )
  
  if  test $? != 0 ; then
      echo "Error processing $dir"
      exit $? 
  fi

done

echo 'Finished running autotools.  Run ./configure next.'


