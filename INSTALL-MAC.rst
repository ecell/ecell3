Installing on Mac OS X
======================

1. Install xcode command line tools

-  open xcode
-  open **references** from menu
-  select **downloads** icon and **components** tab
-  install **command line tools**

2. Install homebrew and C++ dependencies

   ::

       ruby -e "$(curl -fsSkL raw.github.com/mxcl/homebrew/go)"
       brew install autoconf automake libtool gsl boost149

3. Install pip and Python dependencies

   ::

       sudo easy_install pip
       sudo pip install ply

   download and install pygtk Mac OS X package from
   http://sourceforge.net/projects/macpkg/files/PyGTK/2.24.0/

4. Install ecell

   ::

       git clone git://github.com/ecell/ecell3
       cd ecell3
       git checkout -t remotes/origin/gcc47
       sh autogen.sh
       LDFLAGS=-L/usr/local/opt/boost149/lib CPPFLAGS=-I/usr/local/opt/boost149/include ./configure --with-boost-python-libname=boost_python-mt
       make
       make install # do NOT use sudo, sudo command changes owner of /usr/loca/bin[, etc, lib].

5. configure PYTHONPATH

   edit your .bashrc or .zshrc

   ::

       export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH


