#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#		This file is part of E-CELL Model Editor package
#
#				Copyright (C) 1996-2003 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Dini Karnaga, Sylvia Tarigan, Thaw Tint <polytech@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Dini Karnaga, Sylvia Tarigan, Thaw Tint' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


import gtk

from ModelEditor import *
from ListWindow import *
import os
import os.path
import string


class LayoutManagerWindow( ListWindow ):


	def __init__( self, theModelEditor ):
               
		"""
		in: ModelEditor theModelEditor
		returns nothing
		"""
		self.theModelEditor=theModelEditor
		# init superclass
		ListWindow.__init__( self, theModelEditor )


	def openWindow( self ):
		"""
		in: nothing
		returns nothing
		"""

		# superclass openwindow
		ListWindow.openWindow( self )

		# add signal handlers


		self.addHandlers({ 
				'on_CreateButton_clicked' : self.__create_layout,\
				'on_DeleteButton_clicked' : self.__delete_layout,\
				'on_CopyButton_clicked' : self.__copy_layout,\
				'on_ShowButton_clicked' : self.__show_layout,\
				 })


	def move( self, xpos, ypos ):
		ListWindow.move(self,xpos,ypos)


        def deleted( self, *args ):
		
		ListWindow.deleted( self, args )
		self.theModelEditor.toggleOpenLayoutWindow(False)

	def __create_layout( self, *args ):
		print 'Create Button pressed'

	def __delete_layout( self, *args ):
		print 'Delete Button pressed'

	def __copy_layout( self, *args ):
		print 'Copy Button pressed'
	def __show_layout( self, *args ):
		print 'Show Button pressed' 


