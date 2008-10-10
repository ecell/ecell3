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
# Author: Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
#

import os

import gtk
import gtk.gdk
import gtk.glade

from Window import Window

class Pane( Window ):
    def __init__( self, aGladeFile = None, aRootWidgetName = None ):
        """
        constructor
        aGladeFile   -- a glade file name (str)
        aRootWidgetName -- root widget id
        """
        # calls superclass's constructor
        Window.__init__(
            self, aGladeFile,
            aRootWidgetName == None and 'top_frame' or aRootWidgetName )
