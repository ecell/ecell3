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
#'Programming: Sylvia Tarigan' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from Utils import *
import gtk
import gobject

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from EntityEditor import *
from VariableReferenceEditorComponent import *
from Constants import *
from LayoutManager import *
from Layout import *
from EditorObject import *


class ObjectEditorWindow :

	def __init__( self, aModelEditor, aLayoutName, anObjectId ):
		"""
		sets up a modal dialogwindow displaying 
		either both the EntityEditor and the ShapeProperty
                or the ConnectionObjectEditorWindow

		""" 
		self.theModelEditor = aModelEditor	
		
		# Create the Dialog
		self.win = gtk.Dialog('Object Editor Window', None)
		self.win.connect("destroy",self.destroy)

		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)


		# Sets title
		self.win.set_title("ObjectEditor")
		
		self.getTheObjectProperty(aLayoutName, anObjectId)
		
		
		if self.theObject.getProperty(OB_HASFULLID):
			
		
	                self.theComponent = EntityEditor( self, self.win.vbox,OB_TYPE_VARIABLE)
			FullId = self.theObject.getProperty(OB_FULLID)
		
			self.theComponent.setDisplayedEntity (FullId)
			
                	#Add the ShapePropertyComponent
	   		aNoteBook=ViewComponent.getWidget(self.theComponent,'editor_notebook')
			aShapeFrame=gtk.Frame()
			aShapeFrame.show()
			aShapeLabel=gtk.Label('ShapeProperty')
			aShapeLabel.show()
			aNoteBook.append_page(aShapeFrame,aShapeLabel)
		
			self.theComponent.theShapeProperty = ShapePropertyComponent( self.theComponent.theParentWindow, aShapeFrame )
			self.theComponent.theShapeProperty.setDisplayedShapeProperty(self.theObject,FullId,self.theObjShapeType, self.theObjWidth, self.theObjHeight)
			self.theComponent.update()
			
		
              	else:
			self.theShapeProperty=ShapePropertyComponent( self, self.win.vbox )



		self.win.show_all()
		self.theModelEditor.toggleObjectEditorWindow(True,self)




	# ==========================================================================
	def getTheObjectProperty(self,aLayoutName, anObjectId):
		self.theLayout =self.theModelEditor.theLayoutManager.getLayout(aLayoutName)
		self.theObject = self.theLayout.getObject(anObjectId)
		self.theObjWidth = self.theObject.theShape.width
		self.theObjHeight = self.theObject.theShape.height 
		self.theObjShapeType = self.theObject.getProperty(OB_SHAPE_TYPE)
		
		
	# ==========================================================================
	def displayObjectEditorWindow(self,aLayoutName, anObjectId):
		self.getTheObjectProperty(aLayoutName, anObjectId)
		if self.theObject.getProperty(OB_HASFULLID):
			FullId = self.theObject.getProperty(OB_FULLID)
		        self.theComponent.setDisplayedEntity(FullId)
			self.theComponent.theShapeProperty.setDisplayedShapeProperty(self.theObject,FullId,self.theObjShapeType, self.theObjWidth, self.theObjHeight)
		else:
			self.theShapeProperty=ShapePropertyComponent( self, self.win.vbox )
			
			


	# ==========================================================================
	def return_result( self ):
		"""Returns result
		"""

		return self.__value


	# ==========================================================================
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		
		self.win.destroy()
		
	        self.theModelEditor.toggleObjectEditorWindow(False,None)
		
		

		
	




