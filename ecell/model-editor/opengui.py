#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
import os
import sys
import pygtk
pygtk.require('2.0')
import gtk
import getopt

import ModelEditor

usageString = '''
%(appname)s -- model editor for ecell3

Usage:
     %(appname)s [-f EMLFILE] [-h]

     -f   : Specify a EML file to load on startup.
     -h   : Prints this help.

Configurations:
    If ECELL3_DM_PATH environment variable is set to a colon (%(pathsep)s)
    separated directory path, it tries to find dynamic modules within the
    locations referred to by it.

    Example: 
      ECELL3_DM_PATH=/home/user/dm%(pathsep)s/home/user/dm_other %(appname)s

Authors:
    Gabor Bereczki <gabor.bereczki@talk21.com>
    Satya Nanda Vel Arjunan
    Masahiro Sugimoto
    Sylvia Kinta
    Dini Karnaga
    Thaw Tint
          
''' % { 'appname': os.path.basename( sys.argv[ 0 ] ), 'pathsep': os.pathsep }

def main():
    aFileName = None
    if len( sys.argv ) > 1:
        try:
            optstrings = getopt.getopt( sys.argv[1:], 'f:h' )[0]
        except:
            print usageString
            sys.exit( -1 )
            
        if len( optstrings ) > 0:
            for anOpt in optstrings:
                if anOpt[0] == '-h':
                    print usageString
                    sys.exit(0)
                elif anOpt[0] == '-f':
                    aFileName = anOpt[1]
            if aFileName == None:
                print usageString
                sys.exit( -1 )

    g=ModelEditor.ModelEditor( aFileName )
    gtk.main()

