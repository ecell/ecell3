#!/usr/bin/env python
from Utils import *
import gtk
import gobject

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from ShapePropertyComponent import *
from LinePropertyEditor import *
from LayoutManager import *
from Layout import *
from EditorObject import *
from LayoutCommand import *
from EntityCommand import *


class ConnectionObjectEditorWindow:
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aModelEditor, aLayoutName, anObjectId ):
		"""
		sets up a modal dialogwindow displaying 
		the VariableReferenceEditor and the LineProperty
             
		""" 
		self.theModelEditor = aModelEditor	
		
		# Create the Dialog
		self.win = gtk.Dialog('ConnectionObject' , None)
		self.win.connect("destroy",self.destroy)

		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)

		# Sets title
		self.win.set_title('ConnectionObjectEditor')
		iconPixbuf = None
		import os
		iconPixbuf = gtk.gdk.pixbuf_new_from_file(os.environ['MEPATH'] \
				+ os.sep + 'glade' + os.sep + "modeleditor.ico")
		self.win.set_icon(iconPixbuf)
		
		
		self.getTheObject(aLayoutName, anObjectId)
		self.theComponent = VariableReferenceEditorComponent( self, self.win.vbox,self.theLayout,self.theObject)
		self.theComponent.setDisplayedVarRef(self.theLayout,self.theObject)
		
		self.win.show_all()
		self.update()
		self.theModelEditor.toggleConnObjectEditorWindow(True,self)
		

		

	#########################################
	#    Private methods			#
	#########################################

	def setDisplayConnObjectEditorWindow(self,aLayoutName, anObjectId):
		self.getTheObject( aLayoutName, anObjectId)
		self.theComponent.setDisplayedVarRef(self.theLayout,self.theObject)
		self.update()
	def getTheObject(self,aLayoutName, anObjectId):
		self.theLayout =self.theModelEditor.theLayoutManager.getLayout(aLayoutName)
		self.theObjectId = anObjectId
		self.theObject = self.theLayout.getObject(self.theObjectId)
		
	def modifyConnObjectProperty(self,aPropertyName, aPropertyValue):
		aCommand = None
		if  aPropertyName == OB_FILL_COLOR :
			# create command
			aCommand=SetObjectProperty(self.theLayout,self.theObjectId,aPropertyName, aPropertyValue )  
			if aCommand != None:
				self.theLayout.passCommand( [aCommand] )
		if  aPropertyName == OB_SHAPE_TYPE :
			# create command
			aCommand=SetObjectProperty(self.theLayout,self.theObjectId,aPropertyName, aPropertyValue ) 
			if aCommand != None:
				self.theLayout.passCommand( [aCommand] )


	def update(self, aType = None, aFullID = None):
	
		self.theComponent.update()
		
		
		

	# ==========================================================================
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		
		self.win.destroy()

		self.theModelEditor.toggleConnObjectEditorWindow(False,None)
		

		
		

		

	


	






