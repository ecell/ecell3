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
from LayoutCommand import *
from EntityCommand import *
from ResizeableText import *

class ObjectEditorWindow :

	def __init__( self, aModelEditor, aLayoutName, anObjectId ):
		"""
		sets up a modal dialogwindow displaying 
		the EntityEditor and the ShapeProperty
             
		""" 
		self.theModelEditor = aModelEditor	
		self.isBoxShow=False
		self.isFrameShow=False
		self.attBox=None
		self.frameBox=None # OB_HAS_FULLID=False
		
		# Create the Dialog
		self.win = gtk.Dialog('Object Editor Window', None)
		self.win.connect("destroy",self.destroy)

		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)


		# Sets title
		self.win.set_title("ObjectEditor")
		aPixbuf16 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                                os.sep + "glade" + os.sep + "modeleditor.png")
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                                os.sep + "glade" + os.sep + "modeleditor32.png")
		self.win.set_icon_list(aPixbuf16, aPixbuf32)

		self.theComponent=None
		self.theShapeProperty=None
		
		self.getTheObject(aLayoutName, anObjectId)
		self.theEntityType = self.theObject.getProperty(OB_TYPE)
		if self.theObject.getProperty(OB_HASFULLID):
	                self.createComponent()
			self.isBoxShow=True
			self.attBox=self.getAttachmentBox()

              	elif self.theObject.getProperty(OB_TYPE)== OB_TYPE_TEXT:
			self.theShapeProperty=ShapePropertyComponent( self, self.win.vbox )
			self.isFrameShow=True
			self.frameBox=self.getAttachmentFrame()
			

		self.win.show_all()

		if self.theObject.getProperty(OB_HASFULLID):
			#self.theLastFullID = self.theObject.getProperty(OB_FULLID)
			self.selectEntity( [self.theObject] )
		self.update()

		
		self.theModelEditor.toggleObjectEditorWindow(True,self)	

	def createComponent(self):
		self.theComponent = EntityEditor( self, self.win.vbox,self.theEntityType,self )
		self.theComponent.setDisplayedEntity (self.theObject.getProperty(OB_FULLID))
		self.theShapeProperty=self.theComponent.theShapeProperty
		self.theComponent.update()
		

	# ==========================================================================
	def getTheObject(self,aLayoutName, anObjectId):
		self.theLayout =self.theModelEditor.theLayoutManager.getLayout(aLayoutName)
		self.theObjectId = anObjectId
		self.theObject = self.theLayout.getObject(self.theObjectId)
		

	# ==========================================================================
	def modifyObjectProperty(self,aPropertyName, aPropertyValue):
		aCommand = None
		if  aPropertyName == OB_OUTLINE_COLOR  or aPropertyName == OB_FILL_COLOR :
			# create command
			aCommand=SetObjectProperty(self.theLayout,self.theObjectId,aPropertyName, aPropertyValue )  
			if aCommand != None:
				self.theLayout.passCommand( [aCommand] )
			
		
		elif aPropertyName == OB_FULLID:
			if self.theObject.getProperty(OB_HASFULLID):
				
				aCommand = RenameEntity( self.theModelEditor, self.theObject.getProperty(OB_FULLID), aPropertyValue )
				if aCommand.isExecutable():
					self.theModelEditor.doCommandList( [ aCommand ] )
				self.selectEntity( [self.theObject] )
							

			else:
				pass
		
		elif  aPropertyName == OB_DIMENSION_Y  :
			objHeight = self.theObject.getProperty(OB_DIMENSION_Y)
			deltaHeight = objHeight-aPropertyValue
			maxShiftPos= self.theObject.getMaxShiftPos(DIRECTION_DOWN)
			maxShiftNeg= self.theObject.getMaxShiftNeg(DIRECTION_DOWN) 
			if deltaHeight>0:
				if  maxShiftNeg > deltaHeight:
					# create command
					aCommand=ResizeObject(self.theLayout,self.theObjectId,0,-deltaHeight, 0, 0 )
					self.theLayout.passCommand( [aCommand] )
				else:
					self.updateShapeProperty()
			elif deltaHeight<0:
				if  maxShiftPos > -deltaHeight:
					aCommand=ResizeObject(self.theLayout,self.theObjectId,0, -deltaHeight, 0, 0 )
					self.theLayout.passCommand( [aCommand] )
				
			self.update()
		
		elif  aPropertyName == OB_DIMENSION_X :
		 	objWidth = self.theObject.getProperty(OB_DIMENSION_X)

			deltaWidth = objWidth-aPropertyValue
			maxShiftPos= self.theObject.getMaxShiftPos(DIRECTION_RIGHT)
			maxShiftNeg= self.theObject.getMaxShiftNeg(DIRECTION_RIGHT) 
			if deltaWidth>0:
				if  maxShiftNeg > deltaWidth:
					# create command
					aCommand=ResizeObject(self.theLayout,self.theObjectId,0, 0, 0, -deltaWidth )
					self.theLayout.passCommand( [aCommand] )
				else:
					self.updateShapeProperty()

			elif deltaWidth<0:
				if  maxShiftPos > -deltaWidth:
					# create command
					aCommand=ResizeObject(self.theLayout,self.theObjectId,0, 0,0, -deltaWidth )
					self.theLayout.passCommand( [aCommand] )
				else:
					self.updateShapeProperty()
		
	# ==========================================================================
	def getAttachmentBox(self):
		childs=self.win.vbox.get_children()
		for obj in childs:
			if obj.get_name()=='attachment_box':
				return obj

	def setAttachmentBox(self,action):
		if self.attBox==None:
			return
		if action=='hide':
			self.attBox.hide_all()
			self.isBoxShow=False
		else:
			self.attBox.show_all()
			self.isBoxShow=True

	def getAttachmentFrame(self):
		childs=self.win.vbox.get_children()
		for obj in childs:
			if obj.get_name()=='attachment_frame':
				return obj

	def setAttachmentFrame(self,action):
		if self.frameBox==None:
			return
		if action=='hide':
			self.frameBox.hide_all()
			self.isFrameShow=False
		else:
			self.frameBox.show_all()
			self.isFrameShow=True
	# ==========================================================================

	def setDisplayObjectEditorWindow(self,aLayoutName, anObjectId):
		self.getTheObject( aLayoutName, anObjectId)
		if self.theObject.getProperty(OB_HASFULLID):
			self.theLastFullID = self.theObject.getProperty(OB_FULLID)
			if self.theComponent==None:
				self.createComponent()
				self.attBox=self.getAttachmentBox()
			else:
				self.setAttachmentBox('show')
			if self.isFrameShow:
				self.setAttachmentFrame('hide')

			self.theComponent.setDisplayedEntity (self.theLastFullID)
			self.theShapeProperty=self.theComponent.theShapeProperty
			self.theComponent.update()
			
		elif self.theObject.getProperty(OB_TYPE)== OB_TYPE_TEXT:

			# an OB_TYPE_TEXT is being clicked
			if self.frameBox==None:
				self.theShapeProperty=ShapePropertyComponent( self, self.win.vbox )
				self.frameBox=self.getAttachmentFrame()
				self.isFrameShow=True
			else:
				self.setAttachmentFrame('show')
			if self.isBoxShow:
				self.setAttachmentBox('hide')	

			
			
		self.updateShapeProperty()


	# ==========================================================================
	
	def update(self, aType = None, aFullID = None):

		if self.theObject !=None:
			anObjectID = self.theObject.getID()
			existObjectList = self.theLayout.getObjectList()
			if anObjectID not in existObjectList:
				self.theObject =None
				self.theComponent.setDisplayedEntity (None)
		
			else:		
				if self.theObject.getProperty(OB_HASFULLID):
					self.theLastFullID = self.theObject.getProperty(OB_FULLID)
					if not self.theModelEditor.getModel().isEntityExist(self.theLastFullID):
						self.theLastFullID = None
					self.updatePropertyList()
				else:
					self.theLastFullID = None
					self.updatePropertyList()
		self.updateShapeProperty()


	
	def updatePropertyList ( self, aFullID = None ):
		"""
		in: anID where changes happened
		"""
		# get selected objectID
		propertyListEntity = self.theComponent.getDisplayedEntity()
		# check if displayed fullid is affected by changes

		if propertyListEntity != self.theLastFullID:
			self.theComponent.setDisplayedEntity ( self.theLastFullID )
		else:
			self.theComponent.update()



		

	def updateShapeProperty(self):
		self.theShapeProperty.setDisplayedShapeProperty(self.theObject)

	def selectEntity(self,anEntityList):
		if type(anEntityList) == type(""):
			return
		self.theLastObject = anEntityList[0]
		if self.theObject.getProperty(OB_HASFULLID):
			self.theLastFullID = self.theObject.getProperty(OB_FULLID)
			if not self.theModelEditor.getModel().isEntityExist(self.theLastFullID):
				self.theLastFullID = None
		self.theComponent.setDisplayedEntity (self.theLastFullID)
		self.theComponent.update()

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
		
		

		
	




