#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#


import sys

from ecell.ecssupport import *
from ecell.ui.osogo.Window import *

class PluginWindow( Window ):
    """has some plugin functions
    """

    def __init__( self, aDirname, aPluginManager, aRoot=None ):
        """Constructor
        aDirname        -- a directory name (str:absolute path/relative path)
        aPluginManager  -- a reference to PluginManager (an instance of PluginManager)
        aRoot           -- a root property (str)
        """

        # creates glade file name (str)
        aGladeFile = os.path.join( aDirname , self.__class__.__name__ + ".glade" )

        # calls superclass's constructor
        Window.__init__( self, aGladeFile, aRoot )

        self.thePluginManager = aPluginManager  # PluginManager

    def openWindow( self ):
        """openWindow
        Returns None
        """

        # calls superclass's method
        Window.openWindow( self )

    def update( self ):
        """(Abstract method)
        update this window
        Returns None
        """

        import inspect
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + 'must be implemented in subclass')

    def exit( self, *arg ):
        """remove this window from PluginManager
        Returns None
        """

        self.thePluginManager.removeInstance( self )
    
    def close( self ):
        """ closes pluginwindow """
        self.exit( None )	

    def getParent( self ):
       if self.theParent == None:
           return self
       return self.theParent
       
# end of PluginWindow

