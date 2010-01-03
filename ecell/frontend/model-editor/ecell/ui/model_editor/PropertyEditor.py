#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
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

import os
import os.path

import gtk
import gobject

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.ViewComponent import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.MEVariableReferenceEditor import *
from ecell.ui.model_editor.StepperChooser import *
from ecell.ui.model_editor.MultiLineEditor import *
from ecell.ui.model_editor.NestedListEditor import *

class PropertyEditor:

    def __init__( self, aPropertyName, aPropertyType, aPropertyValue, aParentWindow ):
        """
        sets up a modal dialogwindow displaying 
        either a MultiLineEditor, NestedListEditor
        MEVariableReferenceEditor or StepperChooser
        """
        self.theModelEditor = aParentWindow.theModelEditor
        self.theParentEntity = aParentWindow.theDisplayedEntity
        self.theType = aParentWindow.theType
        self.thePropertyType = aPropertyType
        self.thePropertyValue = aPropertyValue
        self.thePropertyName = aPropertyName

        # Sets the return number
        self.__value = None

        # Create the Dialog
        self.win = gtk.Dialog(aPropertyName, None, gtk.DIALOG_MODAL)
        self.win.connect("destroy",self.destroy)

        # Sets size and position
        self.win.set_border_width(2)
        self.win.set_default_size(300,75)
        self.win.set_position(gtk.WIN_POS_MOUSE)


        # Sets title
        self.win.set_title("PropertyEditor")
        #aVbox = gtk.Vbox()
        #self.win.vbox.pack_start(aVbox)
        if self.thePropertyName == ME_PROCESS_VARREFLIST:
            self.theComponent = MEVariableReferenceEditor( self, self.win.vbox )
        elif aPropertyName == ME_STEPPERID:
            self.theComponent = StepperChooser( self, self.win.vbox )
        elif self.thePropertyType == DM_PROPERTY_MULTILINE:
            self.theComponent = MultiLineEditor( self, self.win.vbox )
        elif self.thePropertyType == DM_PROPERTY_POLYMORPH:
            self.theComponent = NestedListEditor( self, self.win.vbox )


    
        # appends ok button
        ok_button = gtk.Button("  OK  ")
        self.win.action_area.pack_start(ok_button,False,False,)
        ok_button.set_flags(gtk.CAN_DEFAULT)
        ok_button.grab_default()
        ok_button.show()
        ok_button.connect("clicked",self.oKButtonClicked)


        # appends cancel button
        cancel_button = gtk.Button(" Cancel ")
        self.win.action_area.pack_start(cancel_button,False,False)
        cancel_button.show()
        cancel_button.connect("clicked",self.cancelButtonClicked)   

        self.win.show_all()
        gtk.main()



    # ==========================================================================
    def oKButtonClicked( self, *arg ):
        """If OK button clicked or the return pressed, this method is called.
        """

        # sets the return number
        self.__value = self.theComponent.getValue()
        if self.__value != None:
            self.destroy()


    # ==========================================================================
    def cancelButtonClicked( self, *arg ):
        """If Cancel button clicked or the return pressed, this method is called.
        """

        # set the return number
        self.__value = None
        self.destroy()
    

    # ==========================================================================
    def return_result( self ):
        """Returns result
        """

        return self.__value


    # ==========================================================================
    def destroy( self, *arg ):
        """destroy dialog
        """

        self.win.hide()
        gtk.main_quit()




    
