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

import string
import operator
import os
from ecell.ui.osogo.utils import *
import ecell.util as util
import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoPluginWindow import OsogoPluginWindow

class VariableWindow( OsogoPluginWindow ):
    """VariableWindow
    - displays Value, Concentration and Fixed Property.
    - Value and Fixed Property can be changed.
    """

    def __init__( self, aDirName, aData, aPluginManager ):
        """
        Constructor
        [Note] When the entity has not Value, MolarConc and Fixed Property,
               throws exception (TypeError).
        """

        # calls constructor of superclass
        OsogoPluginWindow.__init__(
            self, aDirName, aData, aPluginManager.theSession )

        # creates EntityStub
        self.theSession = aPluginManager.theSession
        self.theFullIDString = util.createFullIDString( self.getFullID() )
        self.theStub = self.theSession.createEntityStub(
            util.createFullID( self.theFullIDString ) )

        # initializes flags for validation of Property
        aValueFlag = False
        aMolarConcFlag = False
        aFixedFlag = False

        # [1] Checks this entity have Value, MolarConc, Fixed property.
        for aProperty in self.theStub.getPropertyList():
            if aProperty == 'Value':
                    aValueFlag = True
            elif aProperty == 'MolarConc':
                    aMolarConcFlag = True
            elif aProperty == 'Fixed':
                    aFixedFlag = True
        # If this entity does not have 'Value', does not create instance 
        if aValueFlag == False:
            aMessage = "Error: %s does not have \"Value\" property" %self.theFullIDString
            showPopupMessage( OK_MODE, aMessage, 'Error')
            raise TypeError( aMessage )

        # If this entity does not have 'MolarConc', does not create instance 
        if aMolarConcFlag == False:
            aMessage = "Error: %s does not have \"MolarConc\" property" %self.theFullIDString
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            raise TypeError( aMessage )

        # If this entity does not have 'Fixed', does not create instance 
        if aFixedFlag == False:
            aMessage = "Error: %s does not have \"Fixed\" property" %self.theFullIDString
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            raise TypeError( aMessage )


        #  [2] Checks Value and MolarConc is Number
        # If Value is not Number
        if not operator.isNumberType( self.theStub.getProperty('Value') ):
            aMessage = "Error: \"Value\" property is not number" 
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            raise TypeError( aMessage )

        # If MolarConc is not Number
        if not operator.isNumberType( self.theStub.getProperty('MolarConc') ):
            aMessage = "Error: \"MolarConc\" property is not number" 
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            raise TypeError( aMessage )

    def initUI( self ):
        """overwriets superclass's method
        """

        # calls superclass's method
        OsogoPluginWindow.initUI( self )

        # adds handers
        self.addHandlers(
            {
                'on_fix_checkbox_toggled'             : self.changeFixFlag,
                'on_value_spinbutton_activate'        : self.changeValue,
                'on_value_spinbutton_focus_out_event' : self.changeValue,
                'on_value_spinbutton_changed'   : self.changeValueByButton,
                }
            )

        # sets FULLID to label 
        self["id_label"].set_text( self.theFullIDString )
        self.setIconList(
            os.path.join( config.glade_dir, "ecell.png" ),
            os.path.join( config.glade_dir, "ecell32.png" ) )
        # sets value to each entry and fix_checkbox
        self.update()

    def update( self ):
        """overwrites super class's method
        Returns None
        """

        # updates 'Value Fxed' check box
        self['fix_checkbox'].set_active( self.theStub.getProperty('Fixed') )

        # updates value entry
        self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )

        # updates concentration entry
        self['concentration_entry'].set_text( str(self.theStub.getProperty('MolarConc')) )

    def changeFixFlag( self, *arg ):
        """when 'Value Fixed' checkbox is toggled, this method is called.
        Returns None
        """

        # set 'Fixed' property with the status of toggel button
        self.theStub.setProperty( 'Fixed', self['fix_checkbox'].get_active() )

        # updates plugin manager
        self.theSession.updateUI()

    def changeValue( self, *arg ):
        """When enter is pressed on value entry, this method is called.
        Return None
        """

        # gets text
        aText = string.strip( self['value_spinbutton'].get_text() )

        # The following 2 lines are needed for initialize this window.
        # When openWindow is called, this method must be called but value_spinbutton
        # is empty. The validation of it should be ignored.
        if aText == '':
            return None

        # Convert inputted text to number
        try:
            aNumber = string.atof(aText)

        # When it is not number
        except:
            self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )
            # displays confirm window
            showPopupMessage( OK_MODE, "Input number.", 'Error' )

        # When it is number
        else:
            # sets value
            self.theStub.setProperty( 'Value', aNumber )
            # updates all window
            # not only plugin managers, but also EntityWindow
            self.theSession.updateUI()

    def changeValueByButton( self, *arg ):
        """When a button of value spinbutton is pressed, this method is called.
        Return None
        """

        # converts inputted text into number.
        # When it is number, sets it to EntityStub.
        try:
            aText = string.strip( self['value_spinbutton'].get_text() )
            # The following 2 lines are needed for initialize this window.
            # When openWindow is called, this method must be called but value_spinbutton
            # is empty. The validation of it should be ignored.
            if len(aText) == 0:
                return None

            aNumber = string.atof(aText)
            self.theStub.setProperty( 'Value', aNumber )
            self.theSession.updateUI()
        # When it is not number, sets previous value to value spinbutton.
        except:
            self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )

# end of class VariableWindow


