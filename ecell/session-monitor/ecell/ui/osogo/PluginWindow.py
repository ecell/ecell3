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

from SeparativePane import SeparativePane
import constants

class PluginWindow( SeparativePane ):
    def __init__( self, aDirname, aSession, aRootWidgetName = 'top_frame' ):
        """
        Constructor
        aDirname        -- a directory name (str:absolute path/relative path)
        aSession        -- a reference to SessionFacade
        aRootWidgetName -- a root property (str)
        """
        # calls superclass's constructor
        SeparativePane.__init__(
            self,
            os.path.join( aDirname, self.__class__.__name__ + ".glade" ),
            aRootWidgetName )
        self.theSession = aSession

    def setTitle( self, aTitle ):
        if not self.theSession.onPluginWindowTitleChanging( self, aTitle ):
            return False
        aOldTitle = self.theTitle
        if not SeparativePane.setTitle( self, aTitle ):
            return False
        self.theSession.onPluginWindowTitleChanged( self, aOldTitle )
        return True

    def getName( self ):
        return self.__class__.__name__

    def __str__( self ):
        return "Instance of %s (title=%s)" % (
            self.__class__.__name__, self.theTitle )
# end of PluginWindow

