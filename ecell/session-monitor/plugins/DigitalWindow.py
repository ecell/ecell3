#!/usr/bin/env python
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
import operator

from ecell.ecs_constants import *
from ecell.ui.osogo.constants import *
from ecell.ui.osogo.OsogoPluginWindow import OsogoPluginWindow
from ecell.ui.osogo.ConfirmWindow import ConfirmWindow
import ecell.util as util
import ecell.ui.osogo.config as config

class DigitalWindow( OsogoPluginWindow ):
    """show one numerical property """
    def __init__( self, aDirName, aData, aPluginManager ):
        """
        Constructor
        
        aDirName(str)   : directory name that includes glade file
        data            : RawFullPN
        aPluginManager  : the reference to pluginmanager 
        return -> None
        """
        # call constructor of superclass
        OsogoPluginWindow.__init__(
            self, aDirName, aData, aPluginManager.theSession )

        aValue = self.theSession.getEntityProperty( self.getFullPN() )
        if operator.isNumberType( aValue ) == False:
            aMessage = "Error: (%s) is not numerical data" %\
                util.createFullPNString( self.getFullPN() )
            self.theSession.message( aMessage )
            aDialog = ConfirmWindow(0,aMessage,'Error!')
            raise TypeError( aMessage )

    def initUI( self ):
        OsogoPluginWindow.initUI( self )
        aValue = self.theSession.getEntityProperty( self.getFullPN() )
        anAttribute = self.theSession.getEntityPropertyAttributes(
            self.getFullPN() )

        self.addHandlers(
            { 
                'on_value_frame_activate'        :self.inputValue,
                'on_increase_button_clicked'     :self.increaseValue,
                'on_decrease_button_clicked'     :self.decreaseValue,
                }
            )

        aString = str( self.getFullPN()[ID] )
        aString += ':\n' + str( self.getFullPN()[PROPERTY] )
        self["id_label"].set_text( aString )

        # If this property is not settable, sets unsensitive TextEntry and Buttons.
        if anAttribute[SETTABLE] == False:
            self["value_frame"].set_editable( False )
            self["increase_button"].set_sensitive( False )
            self["decrease_button"].set_sensitive( False )
        self.setIconList(
            os.path.join( config.glade_dir, "ecell.png" ),
            os.path.join( config.glade_dir, "ecell32.png" ) )
        self.update()

    def changeFullPN( self, anObject ):
        """
        changeFullPN
        
        anObject(any)   : a dummy object
        return -> None
        """
        OsogoPluginWindow.changeFullPN( self, anObject )
        aString = str( self.getFullPN()[ID] )
        aString += ':\n' + str( self.getFullPN()[PROPERTY] )
        self["id_label"].set_text( aString )

    def update( self ):
        """
        update
        
        return -> None
        """
        aValue = self.theSession.getEntityProperty( self.getFullPN() )
        self["value_frame"].set_text( str( aValue ) )

    def inputValue( self, *arg ):
        """
        inputValue
        
        anObject(any)   : a dummy object
        return -> None
        """
        # gets text from text field.
        aText = string.split(self['value_frame'].get_text())
        if type(aText) == type([]):
            if len(aText) > 0:
                aText = aText[0]
            else:
                return None
        else:
            return None

        # Only when the length of text > 0,
        # checks type of text and set it.
        if len(aText)>0:
            # Only when the value is numeric, 
            # the value will be set to value_frame.
            try:
                aValue = string.atof( aText )
                self.setValue( self.getFullPN(), aValue )
            except:
                ConfirmWindow(0,'Input numerical value.')
                aValue = self.getValue( self.getFullPN() )
                self["value_frame"].set_text( str( aValue ) )
            return None
        else:
            return None

    def increaseValue( self, obj ):
        """
        increaseValue
        
        anObject(any)   : a dummy object
        return -> None
        """
        if self.getValue( self.getFullPN() ):
            self.setValue( self.getFullPN(), self.getValue( self.getFullPN() ) * 2.0 )
        else:
            self.setValue( self.getFullPN(), 1.0 )
        
    def decreaseValue( self, obj ):
        """
        decreaseValue
        
        anObject(any)   : a dummy mobject
        return -> None
        """
        self.setValue( self.getFullPN(), self.getValue( self.getFullPN() ) * 0.5 )
