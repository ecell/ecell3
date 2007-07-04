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
#             'Yuusuke Saito'
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os
import string
import sys
import gtk

from Pane import Pane
import constants

class SeparativePane( Pane ):
    def __init__( self, aGladeFile = None, aRootWidgetName = None ):
        # calls superclass's constructor
        Pane.__init__( self, aGladeFile, aRootWidgetName )
        # None if "dock"ed
        self.theOuterFrame = None

    def setTitle( self, aTitle ):
        if not Pane.setTitle( self, aTitle ):
            return False
        if self.theOuterFrame != None:
            self.theOuterFrame.set_title( aTitle )
        return True

    def handleDeleteEvent( self, *args ):
        self.destroy()

    def destroy( self ):
        """
        closes pluginwindow
        """
        if self.theOuterFrame != None:
            self.theOuterFrame.destroy()
            self.theOuterFrame = None
        Pane.destroy( self )

    def _setParentIntn( self, aParent, aParentWidgetName ):
        """
        The underlying widget must be initialized prior to a call to this
        function.
        """
        # XXX: should not be called asynchronously
        if aParent == None:
            if self.theOuterFrame == None:
                if self.theParent != None and \
                   self.theParentWidgetName != None and \
                   self.theRootWidget != None:
                    aParentWidget =  self.theParent[
                        self.theParentWidgetName ]
                    # it is possible that the parent widget is
                    # already destroyed or unmanaged by the parent
                    # window wrapper.
                    if aParentWidget != None:
                        aParentWidget.remove( self.theRootWidget )
        else:
            if self.theOuterFrame != None:
                self.theOuterFrame.remove( self.theRootWidget )
        Pane._setParentIntn( self, aParent, aParentWidgetName )
        if aParent != None and self.theOuterFrame != None:
            self.theOuterFrame.destroy()
        if self.shown:
            self.show()

    def show( self ):
        Pane.show( self )
        if self.theParent == None and self.theOuterFrame == None:
            anOuterFrame = gtk.Window()
            anOuterFrame.set_position( gtk.WIN_POS_MOUSE )
            anOuterFrame.set_icon_list( *self.loadWindowIcons() )
            anOuterFrame.set_title( self.theTitle )
            anOuterFrame.add( self.theRootWidget )
            anOuterFrame.connect(
                'delete-event', lambda w, event: self.handleDeleteEvent() )
            anOuterFrame.show()
            self.theOuterFrame = anOuterFrame

    def present( self ):
        """moves this window to the top of desktop.
        if plugin is on BoardWindow, does nothing.
        """
        if self.theParent == None and self.theOuterFrame != None:
            self.theOuterFrame.present()


