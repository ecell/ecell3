#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito',
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.

import sys
import os
from gtk import *
import traceback

from ecell.ecs_constants import *
from ecell.ecssupport import *
from ecell.ui.osogo.PluginWindow import *

from ecell.ui.osogo.config import *
from ecell.ui.osogo.OsogoUtil import *
from ecell.ui.osogo.ConfirmWindow import *

class OsogoPluginWindow(PluginWindow):
    """OsogoPluginWindow
    This class has the following attribute and methods.

    self.theRawFullPNList : [ FullPN1, FullID2, FullPN3, , , ]
    getFullPNList()       : [ FullPN1, FullPN2, FullPN3, , , ]
    getFullIDList()       : [ FullID1, FullID2, FullID3, , , ]
    getFullPN()           : FullPN1
    [Note]:When the Property of FullPN is wrong, the constructor of subclass
           should throw TypeError. PluginManager will catch this Error,
           display error message and create nothing.    
    """

    # If the window displays multiple FullPN/FullID, theViewType is MULTIPLE
    theViewType = SINGLE  # default


    def __init__( self, dirName, data, pluginManager, rootWidget=None ):
        """Constructor
        dirName        --  a directory name including plugin module
                            (str:absolute path/relative path)
        data           --  a RawFullPNList (RawFullPNList)
        pluginManager  --  a reference to PluginManager (PluginManager)
        rootWidget      --  a root widget (str or None)
        """

        #self.theFullPNListClipBoard = []
        self.theSelectedFullPNIndex = 0

        # calls superclass's constructor
        PluginWindow.__init__( self, dirName, pluginManager, rootWidget )

        self.theSession = self.thePluginManager.theSession 
        self.theRawFullPNList = data

        # sets default title
        self.theTitle = self.__class__.__name__


    def openWindow( self ):
        """overwrites superclass's method
        Returns None
        [Note]:When this is top window, appends 'destroy' signal handler.
        """

        # calls superclass's method
        PluginWindow.openWindow( self )

        # When this is top window, appends 'destroy' signal handler.
        #        if self.theRoot == None:
        topWindow = self[self.__class__.__name__]
        if topWindow != None:
            self[self.__class__.__name__].connect('destroy',self.exit)


    def setRawFullPNList( self, aRawFullPNList ):
        """sets RawFullPNList
        aRawFullPNList  --  a RawFullPNList to be set (RawFullPNList)
        Returns None
        """
        self.theRawFullPNList = aRawFullPNList


    def appendRawFullPNList( self, aRawFullPNList ):
        """appneds RawFullPNList
        aRawFullPNList  --  a RawFullPNList to be appned (RawFullPNList)
        Returns None
        """

        self.theRawFullPNList += aRawFullPNList 

    # ---------------------------------------------------------------
    # getRawFullPNList
    #   - return RawFullPNList
    #
    # return -> RawFullPNList
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def getRawFullPNList( self ):
        return self.theRawFullPNList 

    # ---------------------------------------------------------------
    # getFullPNList
    #   - return FullPNList
    #
    # return -> FullPNList
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def getFullPNList( self ):
        return map( self.supplementFullPN, self.theRawFullPNList )

    # end of getFullPNList


    # ---------------------------------------------------------------
    # theFullIDList
    #   - return FullIDList
    #
    # return -> FullIDList
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def getFullIDList( self ):

        return map( convertFullPNToFullID, self.theRawFullPNList )

    # ---------------------------------------------------------------
    # theFullPN
    #   - return FullPN
    #
    # return -> FullPN
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def getFullPN( self ):
        if len( self.theRawFullPNList ) == 0:
            return None
        return self.supplementFullPN( self.theRawFullPNList[self.theSelectedFullPNIndex] )

    # ---------------------------------------------------------------
    # supplementFullID
    #   - supplements default parameter to FullID
    #   - return the supplemented FullID
    #
    # return -> supplemented FullID
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def supplementFullPN( self, aFullPN ):
        if aFullPN[PROPERTY] != '' :
            return aFullPN
        else :
            if aFullPN[TYPE] == VARIABLE :
                aPropertyName = 'Value'
            elif aFullPN[TYPE] == PROCESS :
                aPropertyName = 'Activity'
            elif aFullPN[TYPE] == SYSTEM :
                aPropertyName = 'Size'
            aNewFullPN = tuple( aFullPN[ 0:3 ] ) + ( aPropertyName, )
            return aNewFullPN

    # end of supplementFullPN


    # ---------------------------------------------------------------
    # exit
    #   - call exit method of superclass 
    #
    # *objects  : dammy element of arguments
    #
    # return -> None
    # This method throws exceptions.
    # ---------------------------------------------------------------
    def exit( self, *objects ):
        # call exit method of superclass 
        PluginWindow.exit(self, *objects)

    def isStandAlone(self):
        """ returns True if plugin is in a separate window
            False if it is on a BoardWindow
        """
        return self.getParent().__class__.__name__.startswith( self.__class__.__name__)
        
    def present( self ):
        """moves this window to the top of desktop.
        if plugin is on BoardWindow, does nothing.
        Returns None
        """

        if self.isStandAlone():
            self[self.__class__.__name__].present()

    def iconify( self ):
        """moves this window to the taskbar.
        When it is on Boardwindow, does nothing.
        Returns None
        """
    
        if self.isStandAlone():
            self[self.__class__.__name__].iconify()

    def move( self, xpos, ypos ):
        """moves this window on the desktop to (xpos,ypos).
        When it is on Boardwindow, does nothing.
        Returns None
        """

        if self.isStandAlone():
            self[self.__class__.__name__].move( xpos, ypos)

    def resize( self, width, heigth ):
        """resizes this window according to width and heigth.
        Returns None
        """
        self[self.__class__.__name__].resize( width, heigth)


# end of OsogoPluginWindow
