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
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito',
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.

import os

from gtk import *
import string
import sys
import traceback
from warnings import *

from ecell.ecs_constants import *
import ecell.util as util

from constants import *
from PluginWindow import *
from utils import *

class OsogoPluginWindow(PluginWindow):
    """
    OsogoPluginWindow
    This class has the following attribute and methods.

    self.theRawFullPNList : [ FullPN1, FullID2, FullPN3, , , ]
    getFullPNList()       : [ FullPN1, FullPN2, FullPN3, , , ]
    getFullIDList()       : [ FullID1, FullID2, FullID3, , , ]
    getFullPN()           : FullPN1
    getFullID()           : FullID1
    [Note]:When the Property of FullPN is wrong, the constructor of subclass
           should throw TypeError. PluginManager will catch this Error,
           display error message and create nothing.    
    """

    # If the window displays multiple FullPN/FullID, theViewType is MULTIPLE
    theViewType = SINGLE  # default

    def __init__( self, dirName, aRawFullPNList, aSession, aRootWidgetName = None ):
        """Constructor
        dirName         -- a directory name including plugin module
                           (str:absolute path/relative path)
        aRawFullPNList  -- a RawFullPNList (RawFullPNList)
        aSession        -- reference to SessionFacade
        aRootWidgetName -- a root widget (str or None)
        """
        PluginWindow.__init__( self, dirName, aSession, aRootWidgetName )

        self.theSelectedFullPNIndex = 0
        self.theRawFullPNList = aRawFullPNList

    def setRawFullPNList( self, aRawFullPNList ):
        """
        sets RawFullPNList
        aRawFullPNList  --  a RawFullPNList to be set (RawFullPNList)
        Returns None
        """

        self.theRawFullPNList = aRawFullPNList

    def appendRawFullPNList( self, aRawFullPNList ):
        """
        appneds RawFullPNList
        aRawFullPNList  --  a RawFullPNList to be appned (RawFullPNList)
        Returns None
        """

        self.theRawFullPNList += aRawFullPNList 

    def getRawFullPNList( self ):
        """
        - return RawFullPNList
        return -> RawFullPNList
        This method can throw an exception.
        """
        return self.theRawFullPNList 

    def getFullPNList( self ):
        """
        getFullPNList
          - return FullPNList
        
        return -> FullPNList
        This method can throw an exception.
        """
        return map( self.supplementFullPN, self.theRawFullPNList )

    def getFullIDList( self ):
        """
        - return FullIDList
        return -> FullIDList
        This method can throw an exception.
        """
        return map( util.convertFullPNToFullID, self.theRawFullPNList )

    def getFullPN( self ):
        """
        - return FullPN
        
        return -> FullPN
        This method can throw an exception.
        """
        if len( self.theRawFullPNList ) <= self.theSelectedFullPNIndex:
            return None

        return self.supplementFullPN(
            self.theRawFullPNList[ self.theSelectedFullPNIndex ] )

    def getFullID( self ):
        """
        theFullID
          - return FullID
        
        return -> FullID
        This method can throw an exception.
        """
        warn( 'DEPRECATED', DeprecationWarning, stacklevel = 2 )
        aFullPN = self.getFullPN()
        return aFullPN != None and aFullPN.fullID

    def supplementFullPN( self, aFullPN ):
        """
        supplementFullID
          - supplements default parameter to FullID
          - return the supplemented FullID
        
        return -> supplemented FullID
        """
        if aFullPN.propertyName != '' :
            return aFullPN
        else:
            if aFullPN[TYPE] == VARIABLE:
                aPropertyName = DEFAULT_VARIABLE_PROPERTY
            elif aFullPN[TYPE] == PROCESS:
                aPropertyName = DEFAULT_PROCESS_PROPERTY
            elif aFullPN[TYPE] == SYSTEM:
                aPropertyName = DEFAULT_SYSTEM_PROPERTY
            return aFullPN.fullID.createFullPN( aPropertyName )

    def getValue( self, aFullPN ):
        """
        getValue from the session.simulator
          - return a value
        
        aFullPN : FullPN
        return -> attribute map 
        This method can throw an exception.
        """
        return self.theSession.getEntityProperty( aFullPN )

    def setValue( self, aFullPN, aValue ):
        """
        - sets value to the session.simulator
        
        aFullPN : FullPN
        aValue  : one element or tuple
        
        return -> None
        """
        return self.theSession.setEntityProperty( aFullPN, aValue )

    def createLogger( self, *objects ):
        """
        createLogger
          - create Logger of theFullPN
        
        *objects : dammy objects
        
        return -> None
        """
        for aFullPN in self.getFullPNList():
            self.theSession.createLogger( aFullPN )

    def changeFullPN( self, anObject ):
        """
        anObject : the FullID that this instance will show
        
        return -> None
        This method can throw an exception.
        """
        index = 0
        for aFullPN in self.getFullPNList():
            aFullPNString = util.createFullPNString( aFullPN )
            if aFullPNString == anObject.get_name():
                break
            index = index + 1

        self.theSelectedFullPNIndex = index
        self.theSession.updateUI()

    def isStandAlone(self):
        """
        returns True if plugin is in a separate window
        False if it is on a BoardWindow
        """
        return self.getParent() == None

    def handleSessionEvent( self, *args ):
        pass
