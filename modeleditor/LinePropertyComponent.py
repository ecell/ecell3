#!/usr/bin/env python


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
#'Programming: Thaw Tint' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from ShapePropertyComponent import *
from LinePropertyComponent import *


class  LinePropertyComponent(ViewComponent):
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aParentWindow, pointOfAttach ):
		
		
		self.theModelEditor = aParentWindow.theModelEditor
		self.theParent  =aParentWindow
		
		ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'LinePropertyComponent.glade' )


		#Add Handlers

		self.addHandlers({
			'on_arw_color_changed'	: self.__change_arw_color,
			'on_arw_style_changed'	: self.__change_arw_style,
			'on_arw_type_changed'	: self.__change_arw_type,
			'on_arw_heads_toggled'	: self.__toggle_arw_heads,
			
			})


		                
		
                
        def close( self ):
		"""
		closes subcomponenets
		"""
		ViewComponent.close(self)

	#########################################
	#    Private methods/Signal Handlers    #
	#########################################


			

		

	def __change_arw_color( self, textString ):
		newClass = (ViewComponent.getWidget(self,'arw_color')).get_text()
		if str(newClass) == 'Black':
			print 'Arrow color set to #00000.'
		if str(newClass) == 'Red':
			print 'Arrow color set to #FF0000.'
		if str(newClass) == 'Green':
			print 'Arrow color set to #00FF00.'
		if str(newClass) == 'Blue':
			print 'Arrow colro set to #0000FF.'


	def __change_arw_style( self, textString ):
		newClass = (ViewComponent.getWidget(self,'arw_style')).get_text()
		if str(newClass) != '':
			print 'Arrow style set to ' + str(newClass) + '.'

	def __change_arw_type( self, textString ):
		newClass = (ViewComponent.getWidget(self,'arw_type')).get_text()
		if str(newClass) != '':
			print 'Arrow type set to ' + str(newClass) + '.'

	def __toggle_arw_heads( self, textString ):
		newClass = (ViewComponent.getWidget(self,'arw_heads')).get_active()
		if str(newClass) == '1':
			print 'Arrow heads are shown.'
		if str(newClass) == '0':
			print 'Arrow heads are hidden.'
		


	






