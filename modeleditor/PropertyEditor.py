

from Utils import *
import gtk
import gobject

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from MEVariableReferenceEditor import *
from StepperChooser import *
from MultiLineEditor import *
from NestedListEditor import *

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
        elif self.thePropertyType == DM_PROPERTY_NESTEDLIST:
            self.theComponent = NestedListEditor( self, self.win.vbox )


    
        # appends ok button
        ok_button = gtk.Button("  OK  ")
        self.win.action_area.pack_start(ok_button,gtk.FALSE,gtk.FALSE,)
        ok_button.set_flags(gtk.CAN_DEFAULT)
        ok_button.grab_default()
        ok_button.show()
        ok_button.connect("clicked",self.oKButtonClicked)


        # appends cancel button
        cancel_button = gtk.Button(" Cancel ")
        self.win.action_area.pack_start(cancel_button,gtk.FALSE,gtk.FALSE)
        cancel_button.show()
        cancel_button.connect("clicked",self.cancelButtonClicked)   

        self.win.show_all()
        gtk.mainloop()



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
        gtk.mainquit()




    
