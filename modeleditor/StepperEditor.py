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
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from PropertyList import *
from ViewComponent import *
from Constants import *
from StepperCommand import *

class StepperEditor(ViewComponent):

	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aParentWindow, pointOfAttach ):

		# call superclass
		ViewComponent.__init__( self,  pointOfAttach, 'attachment_box' , 'ObjectEditor.glade' )
		self.theParentWindow = aParentWindow
		self.updateInProgress = False
		self.theInfoBuffer = gtk.TextBuffer()
		self.theDescriptionBuffer = gtk.TextBuffer()
		self['classname_desc'].set_buffer( self.theDescriptionBuffer )
		self['user_info'].set_buffer( self.theInfoBuffer )

		# add handlers
		self.addHandlers({
				'on_combo-entry_changed' : self.__change_class, \
				'on_editor_notebook_switch_page' : self.__select_page,
				'on_ID_entry_editing_done' : self.__change_name,
				'on_user_info_move_focus' : self.__change_info
				})
		

		# initate Editors
		self.thePropertyList = PropertyList( self.theParentWindow, self['PropertyListFrame'] )
			
		self['vbox4'].remove( self['hbox3'] ) 
		#self['hbox3'] = None

		# make sensitive change class button for process
		self['class_combo'].set_sensitive( gtk.TRUE )

		self.theModelEditor = self.theParentWindow.theModelEditor
		self.setDisplayedStepper( None )



	def close( self ):
		"""
		closes subcomponenets
		"""
		self.thePropertyList.close()
		ViewComponent.close(self)


	def getDisplayedStepper( self ):
		"""
		returns displayed Stepper
		"""
		return self.theDisplayedStepper


	def update ( self ):
		"""
		"""

		# update Name
		self.updateEditor()

		# update propertyeditor

		self.thePropertyList.update()



	def updateEditor( self ):
		self.updateInProgress = True
		editableFlag = gtk.FALSE
		if self.theDisplayedStepper != None:
			editableFlag = gtk.TRUE
		self['ID_entry'].set_sensitive ( editableFlag )
		self['user_info'].set_sensitive( editableFlag )		

		if self.theDisplayedStepper !=None:
			nameText = self.theDisplayedStepper
		else:
			nameText = ''
		self['ID_entry'].set_text( nameText )
		# delete class list from combo
		self['class_combo'].entry.set_text('')
		self['class_combo'].set_popdown_strings([''])
		self['class_combo'].set_sensitive( gtk.FALSE )
		self['class_combo'].set_data( 'selection', '' )
		descText = ''

		if self.theDisplayedStepper != None:
			self['class_combo'].set_sensitive( gtk.TRUE )

			# get class list
			classStore = copyValue ( self.theModelEditor.getDMInfo().getClassList( ME_STEPPER_TYPE) )

		
			# get actual class

			actualclass = self.theModelEditor.getModel().getStepperClassName( self.theDisplayedStepper )
			self['class_combo'].set_popdown_strings(classStore)
		
			# select class
			self['class_combo'].entry.set_text( actualclass )

			self['class_combo'].set_data( 'selection', actualclass )
			descText = self.theModelEditor.getDMInfo().getClassInfo( actualclass, DM_DESCRIPTION )

		self.__setDescriptionText( descText )


		infoText = ''
		if self.theDisplayedStepper != None:
			infoText = self.theModelEditor.getModel().getStepperInfo( self.theDisplayedStepper)
		self.__setInfoText( infoText )
		self.updateInProgress = False



	def setDisplayedStepper ( self, selectedID ):
		"""
		"""

		self.theDisplayedStepper = selectedID 

		# sets displayed Stepper for Property Editor
		self.thePropertyList.setDisplayedEntity(ME_STEPPER_TYPE,  self.theDisplayedStepper )
		self.updateEditor()


	def addLayoutEditor( self, aLayoutEditor ):
		pass


	def changeClass( self, newClass ):
		currentClass = self.theModelEditor.getModel().getStepperClassName( self.theDisplayedStepper )
		if currentClass == newClass:
			return
		aCommand = ChangeStepperClass( self.theModelEditor, newClass, self.theDisplayedStepper )
		self.theModelEditor.doCommandList( [ aCommand ] )


	def changeName ( self, newName ):

		aCommand = RenameStepper( self.theModelEditor, self.theDisplayedStepper, newName )
		if aCommand.isExecutable:
			self.theDisplayedStepper = newName
			self.noActivate = True
			self.theModelEditor.doCommandList ( [ aCommand ] )
			self.noActivate = False
			self.theParentWindow.selectStepper( [newName ] )
		else:
			self.theModelEditor.printMessage( "%s cannot be renamed to %s"%(self.theDisplayedStepper, newName ), ME_ERROR )
			self.updateEditor()


	def changeInfo( self, newInfo ):
		oldInfo = self.theModelEditor.getModel().getStepperInfo( self.theDisplayedStepper )
		if oldInfo == newInfo:
			return
		aCommand = SetStepperInfo( self.theModelEditor, self.theDisplayedStepper, newInfo )
		self.theModelEditor.doCommandList( [ aCommand ] )



	#########################################
	#    Private methods/Signal Handlers    #
	#########################################


	
	def __change_class( self, *args ):
		"""
		called when class is to be changed
		"""
		if args[0].get_text() == '':
			return
		if self.updateInProgress:
			return
		newClass = self['class_combo'].entry.get_text()

		self.changeClass( newClass )



	def __select_page( self, *args ):
		"""
		called when editor pages are selected
		"""
		pass


	def __change_name ( self, *args ):
		
		if self.updateInProgress:
			return

		newName = self['ID_entry'].get_text()
		self.changeName( newName )


	def __change_info( self, *args ):
		if self.updateInProgress:
			return
		newInfo = self.__getInfoText()
		self.changeInfo( newInfo )


	def __setDescriptionText( self, textString ):
		self.theDescriptionBuffer.set_text( textString )


	def __getInfoText( self ):
		endIter = self.theInfoBuffer.get_end_iter()
		startIter = self.theInfoBuffer.get_start_iter()
		return self.theInfoBuffer.get_text( startIter, endIter, gtk.TRUE )


	def __setInfoText( self, textString ):
		self.theInfoBuffer.set_text( textString )

