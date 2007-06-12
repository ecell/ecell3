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
#'Design: Dini Karnaga, Sylvia Tarigan, Thaw Tint <polytech@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Dini Karnaga, Sylvia Tarigan, Thaw Tint' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


import gtk

from ModelEditor import *
from ListWindow import *
import os
import os.path
import string


#class LayoutManagerWindow( ListWindow ):
class LinePropertyEditor( ListWindow ):
    def __init__( self, aModelEditor ):
        """
        in: ModelEditor theModelEditor
        returns nothing
        """
        ListWindow.__init__( self, aModelEditor )
        self.initUI()

    def initUI( self ):
        """
        in: nothing
        returns nothing
        """
        # superclass openwindow
        ListWindow.initUI( self )

        # add signal handlers
        self.addHandlers({ 
                'on_CreateButton_clicked' : self.__create_layout,\
                'on_DeleteButton_clicked' : self.__delete_layout,\
                'on_CopyButton_clicked' : self.__copy_layout,\
                'on_ShowButton_clicked' : self.__show_layout,\
                'on_btnSet_clicked' : self.__set_properties,\
                 })

    def deleted( self, *args ):
        ListWindow.deleted( self, args )
        self.theModelEditor.toggleOpenLayoutWindow(False)

    def __create_layout( self, *args ):
        #pass
        self.theModelEditor.signalHandlerCreate()

    def __delete_layout( self, *args ):
        #pass
        self.theModelEditor.signalHandlerDelete()

    def __copy_layout( self, *args ):
        #pass
        self.theModelEditor.signalHandlerCopy()

    def __show_layout( self, *args ):
        #pass
        #self.theModelEditor.signalHandlerShow()
        self.theModelEditor.createLinePropertyEditorWindow()

    def __set_properties( self, *args):
        self.theModelEditor.displayLineProperties("Line Properties ...")


