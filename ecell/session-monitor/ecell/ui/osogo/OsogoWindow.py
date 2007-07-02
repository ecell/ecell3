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
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os

import gtk
import gtk.gdk
import gtk.glade

import config
from Window import *
from ConfirmWindow import *
from OsogoUtil import *

class OsogoWindow(Window):
    """OsogoWindow
    - manages existance status.
    """
    def __init__( self, aGladeFile = None, aRootWidget = None ):
        """constructor
        aSession  -- a reference to Session (Session)
        aGladeFile   -- a glade file name (str)
        """
        # calls superclass's constructor
        Window.__init__( self, aGladeFile, aRootWidget )
        self.theSession = None

    def setSession( self, aSession ):
        # saves a reference to Session
        self.theSession = aSession

    def present( self ):
        """moves this window to the top of desktop.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """
        # When glade file is not loaded yet or already deleted, does nothing
        # calla present() method of Window widget of this window.
        if self.exists():
            self.theRootWidget.present()

    def iconify( self ):
        """
        moves this window to the taskbar.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """
        # If glade file is not loaded yet or already deleted, does nothing
        # calls iconify() method of Window widget of this window.
        if self.exists():
            self.theRootWidget.iconify()

    def move( self, xpos, ypos ):
        """
        moves this window on the desktop to (xpos,ypos).
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """
        # If glade file is not loaded yet or already deleted, does nothing
        if self.exists():
            self.theRootWidget.move( xpos, ypos)

    def resize( self, width, height ):
        """
        resizes this window according to width and heigth.
        When glade file is not loaded yet or already deleted, does nothing.
        Returns None
        """
        # If glade file is not loaded yet or already deleted, do nothing.
        if self.exists():
            self.theRootWidget.resize( width, height )

    def handleDeleteEvent( self, *arg ):
        """
        Called when 'delete_event' signal is dispatched
        (for example, [X] button is clicked )
        """
        # close this window
        self.destroy()

    def initUI( self ):
        """
        overwrite super class's method
        When glade file is not loaded yet or already deleted, calls superclass's
        initUI() method and connects 'delete_event' and self.delete() method.
        Returns None
        """
        Window.initUI(self)
        # connects 'delete_event' and self.handleDeleteEvent() method.
        self.theRootWidget.connect( 'delete_event', self.handleDeleteEvent )
        self.setIconList(
            os.path.join( config.glade_dir, "ecell.png" ),
            os.path.join( config.glade_dir, "ecell32.png" )
            )

    def update( self ):
        """
        Returns None
        """

    def handleSessionEvent( self, event ):
        pass
