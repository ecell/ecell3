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
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#
import gtk

import ModelEditor
from ListWindow import *
import os
import os.path
import string
from Constants import *

class PathwayEditor( ListWindow ):


	def __init__( self, theModelEditor ):
		"""
		in: ModelEditor theModelEditor
		returns nothing
		"""

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
				'on_zoom_in_button_clicked' : self.__zoom_in,\
				'on_zoom_out_button_clicked' : self.__zoom_out,\
				'on_zoom_to_fit_button_clicked' : self.__zoom_to_fit,\
				'on_print_button_clicked' : self.__print,\
				'on_layout_name_entry_activate' : self.__rename_layout,\
				'on_layout_name_entry_editing_done' : self.__rename_layout,\
				'on_selector_button_toggled' : self.__palette_toggled,\
				'on_variable_button_toggled' : self.__palette_toggled, \
				'on_system_button_toggled' : self.__palette_toggled,\
				'on_process_button_toggled' : self.__palette_toggled,\
				'on_text_button_toggled' : self.__palette_toggled,\
				'on_custom_button_toggled' : self.__palette_toggled,\
				'on_search_entry_activate' : self.__search,\
				'on_search_entry_editing_done' : self.__search })


	def update( self, arg1 = None, arg2 = None):
		pass

	def deleted( self, *args ):
		ListWindow.deleted( self, args )
				

	def __zoom_in( self, *args ):
		pass


	def __zoom_out( self, *args ):
		pass

	def __zoom_to_fit( self, *args ):
		pass

	def __print( self, *args ):
		pass

	def __rename_layout( self, *args ):
		pass

	def __palette_toggled( self, *args ):
		pass

	def __search( self, *args ):
		pass
