#!/usr/bin/env python

from Window import *

import string
import sys
from ecell.ecssupport import *

class ViewWindow( Window ):

    theFullPNListClipBoard = []

    def __init__( self, gladefile=None, root=None ):

        self.theGladeFile = gladefile
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

















