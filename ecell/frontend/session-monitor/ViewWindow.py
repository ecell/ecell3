#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

from Window import *

import string
import sys
from config import *
from ecell.ecssupport import *

class ViewWindow( Window ):

    theFullPNListClipBoard = []

    def __init__( self, gladefile=None, rootWidget=None ):
        if gladefile == None:
            gladefile = self.__class__.__name__ + '.glade'
        self.theGladeFile = os.path.join( GLADEFILE_PATH, gladefile )
        self.theRoot = root
        

    def initialize( self ):
        pass


    def update( self ):
        pass


    def copyFullPNList(self, a ):
        ViewWindow.theFullPNListClipBoard = self.theRawFullPNList
        #print 'copy :',
        #print ViewWindow.theFullPNListClipBoard


    def pasteFullPNList(self, a ):
        self.theRawFullPNList = ViewWindow.theFullPNListClipBoard
        self.initialize()
        #print 'paste :',
        #print self.theRawFullPNList


    # overwrite in subclass if needed
    def addFullPNList(self, a ):
        self.theRawFullPNList.extend( ViewWindow.theFullPNListClipBoard )
        #print 'add : ',
        #print self.theRawFullPNList

if __name__ == "__main__":
    def mainLoop():
        gtk.main()

    def main():
        mainLoop()

    main()

















