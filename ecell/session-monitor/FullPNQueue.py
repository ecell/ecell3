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

from config import *
import ecell.Window
import gtk

class FullPNQueue:

    def __init__( self, anAttachmentPoint, aFullPNList ):
        self.backwardQueue = []
        self.forwardQueue = []
        self.theRawFullPNList = aFullPNList
        aWindow = ecell.Window.Window(
            os.path.join( GLADEFILE_PATH, self.__class__.__name__ + '.glade'),
            "hbox1"
            )
        aWindow.openWindow()
        aFrame = aWindow['hbox1']
        self.backButton = aWindow['backbutton']
        self.forwardButton = aWindow['forwardbutton']
        anAttachmentPoint.add( aFrame )
        self.backButton.connect( "clicked", self.__goBack )
        self.forwardButton.connect( "clicked", self.__goForward )
        self.callbackList = []
        self.__updateNavigatorButtons()

    def registerCallback( self, aFunction ):
        self.callbackList.append( aFunction )
        apply( aFunction, [self.theRawFullPNList] )


    def pushFullPNList( self, aRawFullPNList ):
        self.backwardQueue.append( self.__copyList ( self.theRawFullPNList  ) )
        self.forwardQueue = []

        self.__applyFullPNList( aRawFullPNList )
        self.__updateNavigatorButtons()
        
    def getActualFullPNList( self ):
        return self.__copyList( self.theRawFullPNList )

    def __applyFullPNList( self, aRawFullPNList ):
        self.theRawFullPNList = self.__copyList( aRawFullPNList )
        for aFunction in self.callbackList:

            apply( aFunction, [aRawFullPNList] )


    def applyFullPNList( self  ):
        for aFunction in self.callbackList:

            apply( aFunction, [self.theRawFullPNList] )



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
        self.__applyFullPNList( rawFullPNList )
        self.__updateNavigatorButtons()

        
    def __goForward( self, *args ):
        if len( self.forwardQueue ) == 0:
            return
        rawFullPNList = self.forwardQueue.pop()
        self.backwardQueue.append( self.__copyList( self.theRawFullPNList ) )
        self.__applyFullPNList( rawFullPNList )
        self.__updateNavigatorButtons()
        

    def __updateNavigatorButtons( self ):
        if len( self.backwardQueue ) == 0:
            backFlag = False
        else:
            backFlag = True
        if len( self.forwardQueue ) == 0:
            forFlag = False
        else:
            forFlag = True
        self.forwardButton.set_sensitive( forFlag )
        self.backButton.set_sensitive( backFlag )
        

# read buttons fromfile
# create from EntityList
# create from ropertyWindow
# selection functions in propwindow
# treewalker
# search
