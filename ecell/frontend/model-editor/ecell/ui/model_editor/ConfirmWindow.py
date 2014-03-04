#!/usr/bin/env python
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
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

import os

import gtk 

import ecell.ui.model_editor.Config as config
from ecell.ui.model_editor.Constants import *

class ConfirmWindow(gtk.Dialog):
    """This is confirm popup window class.

    OK_MODE        : The window has 'OK' button.
    OK_CANCEL_MODE : The window has 'OK' and 'Cancel' button.

    When OK is clicked, return OK_PRESSED
    When Cancel is clicked or close Window, return CANCEL_PRESSED
    """

    # ==========================================================================
    def __init__(self, aMode, aMessage, aTitle='Confirm', buttonList=[] ):
        """Constructor
        aMode    ---  mode number that is 0(OK) or 1(OK and Cancel).
        aMessage ---  the message that is displayed in the center
                      of this window
        aTitle   ---  the title of this window
        returns 0 if OK pressed, returns -1 if Cancel was pressed
        """

        # Sets the return number
        self.___num = CANCEL_PRESSED

        # Create the Dialog
        self.win = gtk.Dialog(aTitle, None, gtk.DIALOG_MODAL)
        self.win.connect("destroy",self.destroy)

        # Sets size and position
        self.win.set_border_width(2)
        self.win.set_default_size(300,75)
        self.win.set_position(gtk.WIN_POS_MOUSE)
        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, "modeleditor.png") )
        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, "modeleditor32.png") )
        self.win.set_icon_list(aPixbuf16, aPixbuf32)
        self.win.show()

        # Sets title
        # self.win.set_title(aTitle)

        # Sets message
        aMessage = '\n' + aMessage + '\n'
        aMessageLabel = gtk.Label(aMessage)
        self.win.vbox.pack_start(aMessageLabel)
        aMessageLabel.show()
        
        # if custom mode:
        bcounter = 0
        if aMode == CUSTOM_MODE:
            for aButtonName in buttonList:
                self.addButton( aButtonName, bcounter, bcounter==0 )
                bcounter += 1
        elif aMode == OK_MODE:
            self.addButton( "  OK  ", OK_PRESSED, True )
        elif aMode == OKCANCEL_MODE:
            self.addButton( "  OK  ", OK_PRESSED, True )
            self.addButton( "Cancel", CANCEL_PRESSED, False )
        elif aMode == YESNO_MODE:
            self.addButton( "  Yes  ", YES_PRESSED, True )
            self.addButton( "  No  ", NO_PRESSED, False )

        gtk.main()


    # ==========================================================================
    def addButton( self, aName, aNumber, default= False ):
        button = gtk.Button( aName )
        self.win.action_area.pack_start( button,False,False,)
        if default:
            button.set_flags(gtk.CAN_DEFAULT)
            button.grab_default()
        button.show()
        button.connect("clicked",self.buttonClicked, aNumber )
    

    # ==========================================================================
    def buttonClicked( self, *arg ):
        """If OK button clicked or the return pressed, this method is called.
        """

        # sets the return number
        self.___num = arg[-1]
        self.destroy()

    

    # ==========================================================================
    def return_result( self ):
        """Returns result
        """

        return self.___num


    # ==========================================================================
    def destroy( self, *arg ):
        """destroy dialog
        """
        self.win.hide()
        gtk.main_quit()


# ----------------------------------------------------
# Test code
# ----------------------------------------------------
if __name__=="__main__":
    c = ConfirmWindow(1,'hoge\n')



