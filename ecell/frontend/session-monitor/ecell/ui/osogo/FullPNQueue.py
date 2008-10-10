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
import gtk

from Pane import Pane

class FullPNQueue( Pane ):
    def __init__( self, aFullPNList ):
        Pane.__init__( self )
        self.backwardQueue = []
        self.forwardQueue = []
        self.theRawFullPNList = aFullPNList
        self.callbackList = []

    def initUI( self ):
        Pane.initUI( self )
        self.backButton = self.getWidget( 'backbutton' )
        self.forwardButton = self.getWidget( 'forwardbutton' )
        self.backButton.connect( "clicked", self.__goBack )
        self.forwardButton.connect( "clicked", self.__goForward )
        self.__updateNavigatorButtons()

    def registerCallback( self, aFunction ):
        self.callbackList.append( aFunction )
        aFunction( self.theRawFullPNList )

    def pushFullPNList( self, aRawFullPNList ):
        self.backwardQueue.append( self.theRawFullPNList )
        self.forwardQueue = []
        self.__applyFullPNList( aRawFullPNList )
        self.__updateNavigatorButtons()
        
    def getActualFullPNList( self ):
        return self.theRawFullPNList

    def __applyFullPNList( self, aRawFullPNList ):
        self.theRawFullPNList = aRawFullPNList
        for aFunction in self.callbackList:
            aFunction( aRawFullPNList )

    def applyFullPNList( self  ):
        for aFunction in self.callbackList:
            aFunction( self.theRawFullPNList )

    def __goBack(self, *args):
        if len( self.backwardQueue ) == 0:
            return
        rawFullPNList = self.backwardQueue.pop()
        self.forwardQueue.append( self.theRawFullPNList )
        self.__applyFullPNList( rawFullPNList )
        self.__updateNavigatorButtons()

    def __goForward( self, *args ):
        if len( self.forwardQueue ) == 0:
            return
        rawFullPNList = self.forwardQueue.pop()
        self.backwardQueue.append( self.theRawFullPNList )
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
