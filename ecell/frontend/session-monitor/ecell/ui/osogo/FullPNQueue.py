#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import gtk

import ecell.ui.osogo.Window as Window
from ecell.ui.osogo.config import *

class FullPNQueue( object ):
    def __init__( self, anAttachmentPoint, aRawFullPNList=None ):
        self.backwardQueue = []
        self.forwardQueue = []
        if aRawFullPNList is not None:
            self.theRawFullPNList = self.__copyList( aRawFullPNList )
        else:
            self.theRawFullPNList = None
        self.backButton = anAttachmentPoint[0]
        self.forwardButton = anAttachmentPoint[1]
        self.thebackbuttonHandle = self.backButton.connect( "clicked", self.__goBack )
        self.theForwardButtonHandle = self.forwardButton.connect( "clicked", self.__goForward )
        self.callbackList = []
        self.__updateNavigatorButtons()

    def __del__( self ):
        self.backButton.disconnect( self.theBackButtonHandle )
        self.forwardButton.disconnect( self.theForwardButtonHandle )

    def registerCallback( self, aFunction ):
        self.callbackList.append( aFunction )

    def pushFullPNList( self, aRawFullPNList ):
        aRawFullPNList = self.__copyList( aRawFullPNList )
        if self.theRawFullPNList is not None:
            self.backwardQueue.append( self.theRawFullPNList )
        self.theRawFullPNList = aRawFullPNList
        self.forwardQueue = []
        self.applyFullPNList()
        self.__updateNavigatorButtons()
        
    def getActualFullPNList( self ):
        return self.theRawFullPNList

    def applyFullPNList( self ):
        for aFunction in self.callbackList:
            apply( aFunction, [ self.theRawFullPNList ] )

    def __copyList( self, aList ):
        newList = []
        for anItem in aList:
            if type( anItem ) in [type( [] ) , type( () ) ]:
                newList.append( anItem )
            else:
                newList.append( self.__copyList( anItem ) )
        return newList
        
    def __goBack(self, *args):
        if len( self.backwardQueue ) == 0:
            return
        rawFullPNList = self.backwardQueue.pop()
        self.forwardQueue.append( self.__copyList( self.theRawFullPNList ) )
        self.theRawFullPNList = rawFullPNList
        self.applyFullPNList()
        self.__updateNavigatorButtons()
        
    def __goForward( self, *args ):
        if len( self.forwardQueue ) == 0:
            return
        rawFullPNList = self.forwardQueue.pop()
        self.backwardQueue.append( self.__copyList( self.theRawFullPNList ) )
        self.theRawFullPNList = rawFullPNList
        self.applyFullPNList()
        self.__updateNavigatorButtons()

    def __updateNavigatorButtons( self ):
        self.forwardButton.set_sensitive( len( self.forwardQueue ) > 0 )
        self.backButton.set_sensitive( len( self.backwardQueue ) > 0 )

# read buttons fromfile
# create from EntityList
# create from ropertyWindow
# selection functions in propwindow
# treewalker
# search
