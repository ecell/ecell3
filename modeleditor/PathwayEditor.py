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
from PathwayCanvas import *

class PathwayEditor( ListWindow ):


	def __init__( self, theModelEditor, aLayout ):
		"""
		in: ModelEditor theModelEditor
		returns nothing
		"""

		# init superclass
		ListWindow.__init__( self, theModelEditor )
		self.theLayout = aLayout
		self.theModelEditor = theModelEditor

		


	def openWindow( self ):
		"""
		in: nothing
		returns nothing
		"""

		# superclass openwindow
		ListWindow.openWindow( self )

		# add signal handlers

		self.thePathwayCanvas = PathwayCanvas( self, self['pathway_canvas'] )
		self.theLayout.attachToCanvas( self.thePathwayCanvas )

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
		self.update()

		#get Palette Button Widgets
		
		selector = ListWindow.getWidget(self,'selector_button')
		selector.set_active(gtk.TRUE)
		variable = ListWindow.getWidget(self,'variable_button')
		process = ListWindow.getWidget(self,'process_button')
		system = ListWindow.getWidget(self,'system_button')
		custom = ListWindow.getWidget(self,'custom_button')
		text = ListWindow.getWidget(self,'text_button')

			
	
		
		self.theButtonDict={ 'selector':PE_SELECTOR,  'variable':PE_VARIABLE  , 'process':PE_PROCESS, 'system':PE_SYSTEM ,  'custom':PE_CUSTOM , 'text':PE_TEXT}
		self.thePaletteButtonDict={'selector': selector, 'variable' : variable , 'process': process,  'system' : system, 'custom' : custom, 'text':text}
		self.theButtonKeys=self.thePaletteButtonDict.keys().sort()
  
  		# Sets the return PaletteButton value
		self.__CurrPaletteButton = 'selector'
		self.__PrevPaletteButton = None
		self.isFirst=True


	def update( self, arg1 = None, arg2 = None):
		if not self.exists():
			return
		self['layout_name_entry'].set_text( self.theLayout.getName() )


	def deleted( self, *args ):
		# detach canvas from layout
		self.thePathwayCanvas.getLayout().detachFromCanvas()
		self.theModelEditor.thePathwayEditorList.remove(self)
		ListWindow.deleted( self, args )
		if self.theModelEditor.theObjectEditorWindow!=None:
			self.theModelEditor.theObjectEditorWindow.destroy(self)	
		
	def getPathwayCanvas( self ):	
		return self.thePathwayCanvas

	def getPaletteButton(self):
		return self.theButtonDict[self.__CurrPaletteButton]
		 	

	def toggle(self,aName,aStat):
		if aStat:
			self.thePaletteButtonDict[aName].set_active(gtk.TRUE)
		else:
			self.thePaletteButtonDict[aName].set_active(gtk.FALSE)
		
		
	def getLayout( self ):
		return self.theLayout



	############################################################

	#Callback Handlers
	############################################################
	def __zoom_in( self, *args ):
		pass


	def __zoom_out( self, *args ):
		pass

	def __zoom_to_fit( self, *args ):
		pass

	def __print( self, *args ):
		pass

	def __rename_layout( self, *args ):
		if len(self['layout_name_entry'].get_text())>0:
			if self.theModelEditor.theLayoutManager.renameLayout(self.theLayout.getName(),self['layout_name_entry'].get_text()):
				self.theModelEditor.updateWindows()
			else:
				self['layout_name_entry'].set_text(self.theLayout.getName())
		else:
			self['layout_name_entry'].set_text(self.theLayout.getName())

	def __palette_toggled( self, *args ):
		aButtonName=args[0].get_name().split('_')[0]
		if self.isFirst:
			if aButtonName!=self.__CurrPaletteButton:
				self.isFirst=False
				self.toggle(aButtonName,True)	
				self.toggle(self.__CurrPaletteButton,False)	
				self.__CurrPaletteButton=aButtonName
				
			elif aButtonName==self.__CurrPaletteButton:
				self.isFirst=False
				if self.__CurrPaletteButton=='selector':
					self.toggle(self.__CurrPaletteButton,True)
				else:	
					self.toggle(self.__CurrPaletteButton,False)
					self.toggle('selector',True)	
					self.__CurrPaletteButton='selector'
			
		else:
			self.isFirst=True

	
			
	def __search( self, *args ):
		pass

	

	
		
		
