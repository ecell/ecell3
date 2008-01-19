#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from OsogoUtil import *

from OsogoWindow import *
import gobject

from os import *

from ecell.ecssupport import *
from ConfirmWindow import *
import sys
import traceback

# Constant value of this class
PROPERTY_INDEX = 0
VALUE_INDEX = 1
GET_INDEX = 2
SET_INDEX = 3

MAX_STRING_NUMBER = 40

import copy

class StepperWindow(OsogoWindow):
	"""StepperWindow
	- displayes all stepper property
	- user can change each stepper property
	"""

	# ==========================================================================
	def __init__( self, aSession ): 
		"""Constructor
		aSession     ---  the reference of session
		aSession  ---  the reference of Session
		"""

		# calls superclass's constructor
		OsogoWindow.__init__( self, aSession, 'StepperWindow.glade' )

		self.theSession = aSession



	# ==========================================================================
	def openWindow(self):



		OsogoWindow.openWindow(self)

		# initializes attributes
		self.theSelectedStepperID = None  # selected stepperID (str)
		self.theFirstPropertyIter = {}    # key:iter(TreeIter)  value:None
		self.theSelectedPath = {}         # key:aStepperID(str)  value:Path
		self.thePropertyMap = {}

		aListStore = gtk.ListStore( gobject.TYPE_STRING )
		self['stepper_id_list'].set_model( aListStore )
		column=gtk.TreeViewColumn('Stepper',gtk.CellRendererText(),text=0)
		self['stepper_id_list'].append_column(column)

		aPropertyModel=gtk.ListStore( gobject.TYPE_STRING,
					 gobject.TYPE_STRING,
					  gobject.TYPE_STRING,
					   gobject.TYPE_STRING,
					    gobject.TYPE_STRING)
		self['property_list'].set_model(aPropertyModel)
		column=gtk.TreeViewColumn( 'Property',gtk.CellRendererText(),\
					   text=PROPERTY_INDEX )
		column.set_resizable(True)
		self['property_list'].append_column(column)
		column=gtk.TreeViewColumn( 'Value',gtk.CellRendererText(),\
					   text=VALUE_INDEX )
		column.set_resizable(True)
		self['property_list'].append_column(column)
		column=gtk.TreeViewColumn( 'Get',gtk.CellRendererText(),\
					   text=GET_INDEX )
		column.set_resizable(True)
		self['property_list'].append_column(column)
		column=gtk.TreeViewColumn( 'Set',gtk.CellRendererText(),\
					   text=SET_INDEX )
		column.set_resizable(True)
		self['property_list'].append_column(column)
		

		# adds handlers
		self.addHandlers({ \
				'on_stepper_id_list_select' : self.__selectStepperID,  # StepperID list
				'on_property_list_select_row' : self.__selectProperty, # Property list
				'on_update_button_clicked' : self.__updateProperty,    # update button
	  			'on_close_button_clicked' : self.deleted,            # close button
			})


		aModel = self['stepper_id_list'].get_model()
		aModel.clear()			

		aFirstIter = None
		#for aValue in self.theStepperIDList: 
		for aValue in self.theSession.getStepperList():
			anIter = aModel.append()
			if aFirstIter == None:
				aFirstIter = anIter
			aModel.set( anIter, 0, aValue )

		self[self.__class__.__name__].show_all()

		self['stepper_id_list'].get_selection().select_iter(aFirstIter)
		self.__selectStepperID(None)

	def close( self ):
		self.theSelectedPath = {}
		OsogoWindow.close(self)


	# ==========================================================================
	def selectStepperID( self, aStepperID ):
		""" selects StepperID on screen and displays its property list 
			if StepperID exists returns True, else returns False
		"""
		anIter=self['stepper_id_list'].get_model().get_iter_first()
		while True:
			if anIter == None:
				return False
			aTitle = self['stepper_id_list'].get_model().get_value(anIter, 0 )
			if aTitle == aStepperID:
				aPath = self['stepper_id_list'].get_model().get_path ( anIter )
				self['stepper_id_list'].set_cursor( aPath, None, False )
				break
			anIter=self['stepper_id_list'].get_model().iter_next(anIter)
		self.__selectStepperID(  None )
		return False

	

	# ==========================================================================
	def __selectStepperID( self, *arg ):
		"""selects stepper ID
		Return None
		"""

		# When window is not created, does nothing.
		if self.exists() == FALSE:
			return None

		# --------------------------------------------------
		# Creates selected StepperSub 
		# --------------------------------------------------
		iter = self['stepper_id_list'].get_selection().get_selected()[1]
		# aStepperID is selected stepper id
		aStepperID = self['stepper_id_list'].get_model().get_value(iter,0)
		
		# When same StepperID is selected, does nothing
		if self.theSelectedStepperID != None and aStepperID == self.theSelectedStepperID:
			return None
		self.theSelectedStepperID = aStepperID
		self.thePropertyMap = {}
		# aStepperStub is selected StepperStub of selected stepper
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )


		
		# updates property list
		aPropertyModel=self['property_list'].get_model()
		aPropertyModel.clear()

		# creats list of ClassName's row
		aList = [ 'ClassName', ]

		# value
		aClassName = aStepperStub.getClassname( )
		aList.append( str(aClassName) )
		self.thePropertyMap[ 'ClassName'] = str( aClassName )

		# gettable and settable
		aList.append( decodeAttribute( TRUE ) )   # gettable is '+'
		aList.append( decodeAttribute( FALSE ) )  # settable is '-'

		# sets this list to TreeModel
		iter = aPropertyModel.append()
		for i in range(0,4):
			aPropertyModel.set_value(iter,i,aList[i])
					    
		self.theFirstPropertyIter[aStepperID] = iter

		# --------------------------------------------------
		# sets all propertys' row other than ClassName
		# --------------------------------------------------
		for aProperty in aStepperStub.getPropertyList():

			# property
			aList = [ aProperty, ]  # first element

			# gettable and settable
			anAttribute = aStepperStub.getPropertyAttributes( aProperty )
			# value
			if anAttribute[GETABLE] == 0:
				continue
			aValue = aStepperStub.getProperty( aProperty )
			self.thePropertyMap[ aProperty ] = aValue

			aValueString = str( aValue )
			# second element
			aList.append( shortenString( aValueString,\
						     MAX_STRING_NUMBER) )  

			aList.append( decodeAttribute(anAttribute[GETABLE]) )  # third element
			aList.append( decodeAttribute(anAttribute[SETTABLE]) ) # forth element

			# sets this list to TreeModel
			iter = aPropertyModel.append( )
			#anIterListElement = [iter]
			for i in range(0,4):
				aPropertyModel.set_value(iter,i,aList[i])

		self.update()

		if self.theSelectedPath.has_key(aStepperID) == TRUE:
			aPath = self.theSelectedPath[aStepperID]
			self['property_list'].get_selection().select_path(aPath)
		else:
			aPath = (0,)
			self.theSelectedPath[aStepperID] = aPath
			self['property_list'].get_selection().select_path(aPath)

		self.__selectProperty()


	# ==========================================================================
	def selectProperty(self, aPropertyName):
		""" selects PropertyName on screen  
			if PropertyName exists returns True, else returns False
		"""
		anIter=self['property_list'].get_model().get_iter_first()

		while True:
			if anIter == None:
				return False
			aTitle = self['property_list'].get_model().get_value(anIter, PROPERTY_INDEX )
			if aTitle == aPropertyName:
				aPath = self['property_list'].get_model().get_path ( anIter )
				self['property_list'].set_cursor( aPath, None, False )
				break
			anIter=self['property_list'].get_model().iter_next(anIter)

		self.__selectProperty(  None )
		return False


	# ==========================================================================
	def __selectProperty( self, *arg ):
		"""when a property is selected, calls this method.
		updates 
		Returns None
		"""

		# If Window is closed, do nothing.
		if self.exists() == FALSE:
			return None

		# --------------------------------------------------
		# Creates selected StepperSub 
		# --------------------------------------------------
		iter_stepper = self['stepper_id_list'].get_selection().get_selected()[1]
		aStepperID = self['stepper_id_list'].get_model().get_value(iter_stepper,0)
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		# --------------------------------------------------
		# gets selected property name
		# --------------------------------------------------
		iter = self['property_list'].get_selection().get_selected()[1]
		# When nothing is selected, does nothing.
		if iter == None:
			return None
		aPropertyName = self['property_list'].get_model().get_value( iter, PROPERTY_INDEX )
		self.theSelectedPath[aStepperID]=self['property_list'].get_model().get_path(iter)

		# --------------------------------------------------
		# sets value to value_entry
		# --------------------------------------------------
		aValue = None

		# If selected Property is 'ClassName'
		if aPropertyName == 'ClassName':
			aValue = aStepperStub.getClassname()

		# If selected Property is not 'ClassName'
		else:
			aValue = aStepperStub.getProperty( aPropertyName )

		self['value_entry'].set_text( str( aValue ) )

		# --------------------------------------------------
		# when the selected property is settable, set sensitive value_entry
		# when not, set unsensitive value_entry
		# --------------------------------------------------
		if self['property_list'].get_model().get_value( iter, SET_INDEX ) == decodeAttribute(TRUE):
			self['value_entry'].set_sensitive( TRUE )
			self['update_button'].set_sensitive( TRUE )
		else:
			self['value_entry'].set_sensitive( FALSE )
			self['update_button'].set_sensitive( FALSE )

	# ==========================================================================
	def updateProperty(self, aValue):
		""" overwrites selected Property on screen  
		"""
		if self['value_entry'].get_editable():
			self['value_entry'].set_text ( str( aValue ) )
			self.__updateProperty(None)

	# ==========================================================================
	def __updateProperty( self, *arg ):
		"""updates property
		Return None
		"""

		# If Window is closed, do nothing.
		if self.exists() == FALSE:
			return None

		# --------------------------------------------------
		# creates selected StepperSub 
		# --------------------------------------------------
		iter = self['stepper_id_list'].get_selection().get_selected()[1]
		aStepperID = self['stepper_id_list'].get_model().get_value(iter,0)
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		# gets selected property row
		iter = self['property_list'].get_selection().get_selected()[1]

		if iter == None:
			aMessage = 'Select a property.'
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			self['statusbar'].push(1,'property is not selected.')
			return None

		self['statusbar'].pop(1)

		# -----------------------------------------------------------
		# gets a value to update
		# -----------------------------------------------------------
		aValue = self['value_entry'].get_text( )

		# -----------------------------------------------------------
		# get a property name from property list
		# -----------------------------------------------------------
		aPropertyName = self['property_list'].get_model().get_value( iter, PROPERTY_INDEX )

		# When the property value is scalar
		if type(aValue) != list and type(aValue) != tuple:

			# converts value type
			anOldValue = aStepperStub.getProperty( aPropertyName )

			# ---------------------------------------------------
			# checks float and int type of inputted value
			# does not check str. ( not needed )
			# ---------------------------------------------------
			# float
			if type(anOldValue) == float:
				try:
					aValue = string.atof(aValue)
				except:
					# displays confirm window
					anErrorMessage = "Input float value."
					self['statusbar'].push(1,anErrorMessage)
					aDialog = ConfirmWindow(OK_MODE,"Can't set property!\n" + anErrorMessage,'Error!')
					return None

			# int
			if type(anOldValue) == int:
				try:
					aValue = string.atoi(aValue)
				except:
					# displays confirm window
					anErrorMessage = "Input int value."
					self['statusbar'].push(1,anErrorMessage)
					aDialog = ConfirmWindow(OK_MODE,"Can't set property!\n" + anErrorMessage,'Error!')
					return None

			# sets new value
			try:
				aStepperStub.setProperty( aPropertyName, aValue )
			except:

				# displays error message to MessageWindow
				anErrorMessage = string.join( traceback.format_exception(\
				                 sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.message( anErrorMessage )

				# displays confirm window
				anErrorMessage = "See MessageWindow for details."
				self['statusbar'].push(1,anErrorMessage)
				aDialog = ConfirmWindow(OK_MODE,aMessage,"Can't set property!\n" + anErrorMessage)
				return None

			# when to set property is succeeded,
			else:

				# refreshs self['property_list']
				self.update()

				# displays message to status bar
				aMessage = "Property is changed."
				self['statusbar'].push(1,aMessage)

		# ------------------------------------
		# When the property value is tuple
		# ------------------------------------
		else:

			# -------------------------------------------
			# converts value type
			# do not check the type of ecah element.
			# -------------------------------------------
			anOldValue = aStepperStub.getProperty( aPropertyName )
			anIndexOfTuple = string.atoi(aNumber)-1

			# -------------------------------------------
			# create tuple to set
			# -------------------------------------------
			if anIndexOfTuple == 0:
				aNewValue = (aValue,) + aValueTuple[anIndexOfTuple+1:]
			elif anIndexOfTuple == len(aValueTuple)-1:
				aNewValue = aValueTuple[:anIndexOfTuple] + (aValue,)
			else:
				aNewValue = aValueTuple[:anIndexOfTuple] + (aValue,) + aValueTuple[anIndexOfTuple+1:]


			# -------------------------------------------
			# sets new value
			# -------------------------------------------
			try:
				aStepperStub.setProperty( aPropertyName, aNewValue )

			except:

				# displays error message to MessageWindow
				anErrorMessage = string.join( traceback.format_exception(\
				                 sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.message( aErroeMessage )

				# displays confirm window
				anErrorMessage = "See MessageWindow for details."
				aDialog = ConfirmWindow(OK_MODE,aMessage,"Can't set property!\n" + anErrorMessage)
				self['statusbar'].push(1,anErrorMessage)
				return None


	# ==========================================================================
	def update( self ):
		"""overwrites superclass's method
		updates the value of self['property_list']
		"""

		# When this window does not created, does nothing
		if self.exists() == FALSE:
			return None

		# clears message on statusbar.
		self['statusbar'].pop(1)

		# gets stepperID
		iter_stepper = self['stepper_id_list'].get_selection().get_selected()[1]
		aStepperID = self['stepper_id_list'].get_model().get_value(iter_stepper,0)
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		iter = self.theFirstPropertyIter[aStepperID]

		# updates all value of self['property_list']
		while(TRUE):
			iter = self['property_list'].get_model().iter_next(iter)
			if iter == None:
				break
			aProperty = self['property_list'].get_model().get_value(iter,0)
			if type ( self.thePropertyMap[ aProperty ] ) != type( () ):
				if aStepperStub.getPropertyAttributes( aProperty )[GETABLE]:
					aValue = str( aStepperStub.getProperty( aProperty ) )
				else:
					aValue = ''
				self['property_list'].get_model().set_value(iter,1,aValue)

		# updates text
		self.__selectProperty()


# end of StepperWindow



