#!/bin/bash
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

[ ! -e doc/users-manual/users-manual ] && mkdir -p doc/users-manual/users-manual

. ./ecell_version.sh

for dir in . libltdl dmtool ecell
  do 
  if echo | sed -E 2>/dev/null; then
    top_srcdir=`echo $dir | sed -E -e 's#([^./][^/]*|.[^/][^/]*)(/|$)#..\2#g'`
  else
    top_srcdir=`echo $dir | sed -e 's#\([^./][^/]*\|\.[^/][^/]*\)\(/\|$\)#..\2#g'`
  fi
  echo -n "Running autotools for $dir ... "
  (cd $dir; \
  { if [ -r configure.ac.in ]; then echo -n 'configure.ac ' && sed -e "s/@ECELL_VERSION_NUMBER@/$ECELL_VERSION_NUMBER/g" configure.ac.in > configure.ac; fi } && \
  { echo -n 'libtoolize '; $LIBTOOLIZE -c --force --automake; } && \
  { echo -n 'aclocal '; aclocal -I$top_srcdir/m4; } && \
  { echo -n 'autoheader '; autoheader -f ; } && \
  { echo -n 'automake ';  automake -c --add-missing --gnu $am_opt; } && \
  { echo -n 'autoconf '; autoconf; } && \
  echo )
  
  if  test $? != 0 ; then
      echo "Error processing $dir"
      exit $? 
  fi

done

echo 'Finished running autotools.  Run ./configure next.'


