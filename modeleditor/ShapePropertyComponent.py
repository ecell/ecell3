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
#'Programming: Dini Karnaga' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from whrandom import randint
from SystemObject import *


class ShapePropertyComponent(ViewComponent):

	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aParentWindow, pointOfAttach ):
		self.theModelEditor = aParentWindow.theModelEditor
		
		ViewComponent.__init__( self, pointOfAttach, 'attachment_frame', 'ShapePropertyComponent.glade' )

		# the variables
		self.theShapeDic={'Label':None,'Height':None,'Width':0,'Type':0,'Fill Color':None,'Outline Color':None,}
		self.theShapeList=[]
		self.theColor=None
		self.theColorDialog=None
		self.theDrawingArea=None
		self.theColorList=['red', 'yellow', 'green', 'brown', 'blue', 'magenta',
                 'darkgreen', 'bisque1']
		self.theCombo=ViewComponent.getWidget(self,'shape_combo')
		self.theObject=None

		# the handlers dictionary
		self.addHandlers({ 
			'on_EntityLabel_activate' : self.__EntityLabel_displayed,\
			'on_EntityLabel_focus_out' : self.__EntityLabel_displayed,\
			'on_ComboEntry_changed' : self.__ShapeType_displayed,\
			'on_EntityWidth_activate' : self.__EntityWidth_displayed,\
			'on_EntityWidth_focus_out' : self.__EntityWidth_displayed,\
			'on_EntityHeight_activate' : self.__EntityHeight_displayed,\
			'on_EntityHeight_focus_out' : self.__EntityHeight_displayed,\
			'on_FillButton_clicked' : self.__ColorFillSelection_displayed,\
			'on_OutlineButton_clicked' : self.__ColorOutSelection_displayed
			})
		
			

	#########################################
	#    Signal Handlers                    #
	#########################################
	def __EntityLabel_displayed( self, *args ):
		aShapeLabel=(ViewComponent.getWidget(self,'ent_label')).get_text()
		self.__CheckDictionaryValue('Label',aShapeLabel)

	def __EntityWidth_displayed( self, *args ):
		aShapeWidth=(ViewComponent.getWidget(self,'ent_width')).get_text()
		if self.checkNumeric(aShapeWidth):
			self.__CheckDictionaryValue('Width',aShapeWidth)
		else:
			ViewComponent.getWidget(self,'ent_width').set_text('0')

	def __EntityHeight_displayed( self, *args ):
		aShapeHeight=ViewComponent.getWidget(self,'ent_height').get_text()
		if self.checkNumeric(aShapeHeight):
			self.__CheckDictionaryValue('Height',aShapeHeight)
		else:
			ViewComponent.getWidget(self,'ent_height').set_text('0')	
	
	def __CheckDictionaryValue(self,aKey,aValue):
		if self.theShapeDic[aKey]!=aValue:
			self.theShapeDic[aKey]=aValue
			self.__ShapeProperty_displayed()

	def __ShapeType_displayed( self, *args ):
		aComboEntryWidget=ViewComponent.getWidget(self,'combo_entry')
		aShapeType=aComboEntryWidget.get_text()
		if aShapeType!='Select':
			self.__CheckDictionaryValue('Type',aShapeType)
		
	def __ColorFillSelection_displayed( self,*args):
		aDialog,aColorSel=self.setColorDialog()
		response=aDialog.run()
		if response==gtk.RESPONSE_OK:
			aColor=aColorSel.get_current_color()
			aDa=ViewComponent.getWidget(self,'da_fill')
			aDa.modify_bg(gtk.STATE_NORMAL,aColor)
			aDialog.destroy()
			self.setHexadecimal(aColor,'Fill Color')
			
		else:
			aDialog.destroy()
			

	def __ColorOutSelection_displayed( self,*args):
		aDialog,aColorSel=self.setColorDialog()
		response=aDialog.run()
		if response==gtk.RESPONSE_OK:
			aColor=aColorSel.get_current_color()
			aDa=ViewComponent.getWidget(self,'da_out')
			aDa.modify_bg(gtk.STATE_NORMAL,aColor)
			aDialog.destroy()
			self.setHexadecimal(aColor,'Outline Color')
			
		else:
			aDialog.destroy()

	def __ShapeProperty_displayed(self,*args):
		keys=self.theShapeDic.keys()
		keys.sort()
		for aKey in keys:
			self.theModelEditor.printMessage(aKey + ':' + str(self.theShapeDic[aKey]),ME_PLAINMESSAGE)
		self.theModelEditor.printMessage('',ME_PLAINMESSAGE)


			
	#########################################
	#    Private methods		        #
	#########################################


	def setDisplayedShapeProperty(self,anObject, selectedID, shapeType, width, height):
		self.theObjectID = selectedID 
		self.theObject=anObject
		self.shapeType = self.getShapeType(shapeType)
		self.shapeWidth = width
		self.shapeHeight = height
		self.populateComboBox()
		self.updateShapeProperty()

	def updateShapeProperty(self):
		if self.theObjectID !=None:
			nameText = self.theObjectID.split(':')[2]
		else:
			nameText = ''
		ViewComponent.getWidget(self,'combo_entry').set_text(self.shapeType)
		ViewComponent.getWidget(self,'ent_width').set_text( str(self.shapeWidth) )
		ViewComponent.getWidget(self,'ent_height').set_text( str(self.shapeHeight ))
		ViewComponent.getWidget(self,'ent_label').set_text( nameText )
		
	def getShapeType(self, shapeType):
		if shapeType ==OB_TYPE_PROCESS:
			shapeType = 'Rectangle'
			self.theShapeList = self.theObject.getAvailableProcessShape()
		if shapeType ==OB_TYPE_VARIABLE:
			shapeType = 'Rounded Rectangle'
			self.theShapeList = self.theObject.getAvailableVariableShape()
		if shapeType ==OB_TYPE_SYSTEM:
			shapeType ='Rectangle'
			self.theShapeList = self.theObject.getAvailableSystemShape()
		if shapeType ==OB_TYPE_TEXT:
			pass
		if shapeType ==OB_TYPE_CONNECTION:
			pass
		return shapeType
	
	def populateComboBox(self):
		# populate list item of combobox
		self.theCombo.set_popdown_strings(self.theShapeList)
		

	def setColorDialog( self, *args ):
		aColor=gtk.gdk.color_parse(self.theColorList[randint (0, 3)])
		aDialog=gtk.ColorSelectionDialog("Select Fill Color")
		aColorSel=aDialog.colorsel
		aColorSel.set_previous_color(aColor)
		aColorSel.set_current_color(aColor)
		aColorSel.set_has_palette(gtk.TRUE)
		return[aDialog,aColorSel]
		
	def setHexadecimal(self,theColor,theColorMode ):
		self.theRed=theColor.red
		self.theGreen=theColor.green
		self.theBlue=theColor.blue
		self.theHex=hex(self.theRed + self.theGreen + self.theBlue)
		self.__CheckDictionaryValue(theColorMode,self.theHex)
	
	def checkNumeric(self, aNumber):
		try:
			eval(aNumber)
			return True
		except:
			return False
	




