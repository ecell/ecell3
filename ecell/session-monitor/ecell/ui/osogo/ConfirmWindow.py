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
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import gtk 
import os
import config

# Constants for ConfirmWindow
OK_MODE = 0
OKCANCEL_MODE = 1

# Constans for result
OK_PRESSED = gtk.RESPONSE_OK
CANCEL_PRESSED = gtk.RESPONSE_CANCEL

class ConfirmWindow(gtk.Dialog):
    """This is confirm popup window class.

    OK_MODE        : The window has 'OK' button.
    OK_CANCEL_MODE : The window has 'OK' and 'Cancel' button.

    When OK is clicked, return OK_PRESSED
    When Cancel is clicked or close Window, return CANCEL_PRESSED
    """

    def __init__( self, aMode, aMessage, aTitle='Confirm' ):
        """Constructor
        aMode    ---  mode number that is 0(OK) or 1(OK and Cancel).
        aMessage ---  the message that is displayed in the center
                      of this window
        aTitle   ---  the title of this window
        """

        # Sets the return number
        self.___num = CANCEL_PRESSED

        # Create the Dialog
        if aMode == OK_MODE:
            aButtonSpec = (
                gtk.STOCK_OK, gtk.RESPONSE_OK
                )
        else:
            aButtonSpec = (
                gtk.STOCK_OK, gtk.RESPONSE_OK,
                gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL
                )
        gtk.Dialog.__init__(
            self, aTitle, None, gtk.DIALOG_MODAL, aButtonSpec )

        # Sets size and position
        self.set_default_size(300,75)
        self.set_position( gtk.WIN_POS_CENTER )
        self.action_area.homogeneous = True

        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.glade_dir, 'ecell.png') )
        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.glade_dir, 'ecell32.png') )
        self.set_icon_list( aPixbuf16, aPixbuf32 )

        # Sets message
        self.vbox.pack_start( gtk.Label( aMessage ) )
