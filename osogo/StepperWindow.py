#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
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
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from OsogoUtil import *

#from Window import *
from OsogoWindow import *
#from gtk import *
#from Numeric import *

from os import *

from ecell.ecssupport import *

#from ecell.DataFileManager import *
#from ecell.ECDDataFile import *

# Constant value of this class
PROPERTY_INDEX = 0
NUMBER_INDEX = 1
VALUE_INDEX = 2
GET_INDEX = 3
SET_INDEX = 4

# ---------------------------------------------------------------
# StepperWindow -> OsogoWindow
#  - displays each stepper's property
#
# ---------------------------------------------------------------
class StepperWindow(OsogoWindow):


	# ---------------------------------------------------------------
	# constructor
	# aSession : the reference of session
	# aMainWindow : the reference of MainWindow
	# ---------------------------------------------------------------
	def __init__( self, aSession, aMainWindow ): 

		OsogoWindow.__init__( self, aMainWindow, 'StepperWindow.glade' )
		self.theSession = aSession
		self.theStepperIDList = ()
		self.theStepperIDListItems = []
		self.theSelectedRowOfPropertyList = -1
		self.theSelectedStepperIDListItem = None

	# end of the __init__


	def openWindow(self):

		OsogoWindow.openWindow(self)
		self.initialize()

		# ---------------------------------------------------------------
		# Adds handlers
		# ---------------------------------------------------------------

		self.addHandlers({ \

				# StepperID list
				'on_stepper_id_list_select'   : self.selectStepperID, 

				# Property list
				'on_property_list_select_row' : self.selectProperty, 

				# update button
				'on_update_button_clicked'    : self.updateProperty, 

				# close button
	  			'on_close_button_clicked'     : self.closeWindow,

			})


	# end of openWindow


	# ---------------------------------------------------------------
	# initializer
	# return -> None
	# ---------------------------------------------------------------
	def initialize( self ):
		self.update()

	# end of initialize

	# ---------------------------------------------------------------
	# selectStepperID
	# objects : ( stepper_id_list[GtkList], selected_item[GtkItem] )
	#
	# return -> None
	# ---------------------------------------------------------------
	def selectStepperID( self, *objects ):

		if self.isShown == gtk.FALSE:
			return None

		self.theSelectedStepperIDListItem = objects[1]

	
	

		for theStepperIDListItem in self.theStepperIDListItems:
			if theStepperIDListItem == objects[1]:
				aStepperID = theStepperIDListItem.get_name()

				self['property_list'].clear()

				aClassName = self.theSession.theSimulator.getStepperClassName( aStepperID )
				aList = [ 'ClassName', ]
				aList.append( '' )
				aList.append( str(aClassName) )
				aList.append( decodeAttribute( TRUE ) )
				aList.append( decodeAttribute( FALSE ) )
				self['property_list'].append( aList )

				for aProperty in self.theSession.theSimulator.getStepperPropertyList( aStepperID ):

					if aProperty == 'PropertyAttributes':
						continue

					elif aProperty == 'PropertyList': # This should be removed.
						continue                  # This should be removed.

					elif aProperty == 'ClassName':    # This should be removed.
						continue                  # This should be removed.

					data =  self.theSession.theSimulator.getStepperProperty( aStepperID, aProperty )

					# ---------------------------
					# When data type is scalar
					# ---------------------------
					if type(data) != type(()):

						#aList = [ aProperty[PROPERTYNAME] ]
						aList = [ aProperty, ]
						aList.append( '' )
						aList.append( str(data) )

						anAttribute = self.theSession.theSimulator.getStepperPropertyAttributes( aStepperID, aProperty )


						aList.append( decodeAttribute(anAttribute[GETABLE]) )
						aList.append( decodeAttribute(anAttribute[SETTABLE]) )

						self['property_list'].append( aList )

					# ---------------------------
					# When data type is tuple
					# ---------------------------
					else:

						aNumber = 0
						for anElement in data:

							#aList = [ aProperty[PROPERTYNAME] ]
							aList = [ aProperty ]

							anAttribute = self.theSession.theSimulator.getStepperPropertyAttributes( aStepperID, aProperty )
							aList.append( `aNumber` )
							aList.append( str(anElement) )

							aList.append( decodeAttribute(anAttribute[GETABLE]) )
							aList.append( decodeAttribute(anAttribute[SETTABLE]) )

							#aList.append( decodeAttribute(aProperty[GETABLE]) )
							#aList.append( decodeAttribute(aProperty[SETTABLE]) )

							self['property_list'].append( aList )

							aNumber = aNumber + 1
							

	# ---------------------------------------------------------------
	# selectProprety
	# *objects : the information of selected row (tuple)
	#
	# return -> None
	# ---------------------------------------------------------------
	def selectProperty( self, *objects ):

		if self.isShown == gtk.FALSE:
			return None

		# ------------------------------------------------------------------
		# gets the number of selected row that is required in updateProprety
		# method.
		# ------------------------------------------------------------------
		self.theSelectedRowOfPropertyList = objects[1]

		# ------------------------------------------------------------------
		# gets value from proprety_list
		# ------------------------------------------------------------------
		aValue = self['property_list'].get_text( objects[1], VALUE_INDEX )

		# ------------------------------------------------------------------
		# sets value to value_entry
		# ------------------------------------------------------------------
		self['value_entry'].set_text( aValue )

		# ------------------------------------------------------------------
		# when the selected property is settable, set sensitive value_entry
		# when not, set unsensitive value_entry
		# ------------------------------------------------------------------
		if self['property_list'].get_text( objects[1], SET_INDEX ) == decodeAttribute(TRUE):
			self['value_entry'].set_sensitive( gtk.TRUE )
			self['update_button'].set_sensitive( gtk.TRUE )
		else:
			self['value_entry'].set_sensitive( gtk.FALSE )
			self['update_button'].set_sensitive( gtk.FALSE )


	# end of selectProprety


	# ---------------------------------------------------------------
	# updateProprety
	# This method doesn't deal with 'ClassName' is selected,
	# but there is no case that this is called with selecting 'ClassName'.
	#
	# return -> None
	# ---------------------------------------------------------------
	def updateProperty( self, *objects ):

		if self.isShown == gtk.FALSE:
			return None

		self['statusbar'].pop(1)

		# ---------------------------------------------------------------------------
		# gets a value to update
		# ---------------------------------------------------------------------------
		aValue = self['value_entry'].get_text( )

		# ---------------------------------------------------------------------------
		# gets a number and property name from property list
		# ---------------------------------------------------------------------------
		aNumber = self['property_list'].get_text( self.theSelectedRowOfPropertyList,
		                                          NUMBER_INDEX )
		aPropertyName = self['property_list'].get_text( self.theSelectedRowOfPropertyList,
		                                                PROPERTY_INDEX )

		# ------------------------------------
		# When the property value is scalar
		# ------------------------------------
		if aNumber == '':

			# ---------------------------------------------------------------------------
			# converts value type
			# ---------------------------------------------------------------------------
			anOldValue = self.theSession.theSimulator.getStepperProperty( 
			                              self.theSelectedStepperIDListItem.get_name(),
			                              aPropertyName )

			try:

				if type(anOldValue) == type(0):
					aValue = string.atoi( aValue )
				elif type(anOldValue) == type(0.0):
					aValue = string.atof( aValue )

			except:
				aErrorMessage = "Error : the type of inputed value is wrong."
				self['statusbar'].push(1,aErrorMessage)
				return None

		
			# ---------------------------------------------------------------------------
			# sets new value
			# ---------------------------------------------------------------------------
			try:
				self.theSession.theSimulator.setStepperProperty( self.theSelectedStepperIDListItem.get_name(),
			    	                                             aPropertyName,
			       	                                          aValue )
			except:

				import sys
				import traceback
				self.printMessage(' can\'t load [%s]' %aFileName)
				aErrorMessage = \
				  string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.printMessage( aErroeMessage )

				aErrorMessage = "Error : refer to the MessageWindow."
				self['statusbar'].push(1,aErrorMessage)
				return None

		# ------------------------------------
		# When the property value is tuple
		# ------------------------------------
		else:

			# ---------------------------------------------------------------------------
			# converts value type
			# ---------------------------------------------------------------------------
			aValueTuple = self.theSession.theSimulator.getStepperProperty( 
			                             self.theSelectedStepperIDListItem.get_name(),
			                             aPropertyName )

			try:

				# Convert value type
				anOldValue = aValueTuple[0]
				if type(anOldValue) == type(0):
					aValue = string.atoi( aValue )
				elif type(anOldValue) == type(0.0):
					aValue = string.atof( aValue )

			except:
				aErrorMessage = "Error : the type of inputed value is wrong."
				self['statusbar'].push(1,aErrorMessage)
				return None


			anIndexOfTuple = string.atoi(aNumber)-1


			# ---------------------------------------------------------------------------
			# create tuple to set
			# ---------------------------------------------------------------------------
			if anIndexOfTuple == 0:
				aNewValue = (aValue,) + aValueTuple[anIndexOfTuple+1:]
			elif anIndexOfTuple == len(aValueTuple)-1:
				aNewValue = aValueTuple[:anIndexOfTuple] + (aValue,)
			else:
				aNewValue = aValueTuple[:anIndexOfTuple] + (aValue,) + aValueTuple[anIndexOfTuple+1:]


			# ---------------------------------------------------------------------------
			# sets new value
			# ---------------------------------------------------------------------------
			try:
				self.theSession.theSimulator.setStepperProperty( self.theSelectedStepperIDListItem.get_name(),
			   	                                              aPropertyName,
			   	                                              aNewValue )
			except:

				import sys
				import traceback
				self.printMessage(' can\'t load [%s]' %aFileName)
				aErrorMessage = \
				  string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.printMessage( aErroeMessage )

				aErrorMessage = "Error : refer to the MessageWindow."
				self['statusbar'].push(1,aErrorMessage)
				return None


		# ---------------------------------------------------------------------------
		# updates property_list
		# ---------------------------------------------------------------------------
		self.selectStepperID( self['stepper_id_list'], self.theSelectedStepperIDListItem )

	# end of updateProprety


	# ---------------------------------------------------------------
	# Updates
	#
	# return -> None
	# ---------------------------------------------------------------
	def update( self ):

		if self.isShown == gtk.FALSE:
			return None

		self['statusbar'].pop(1)

		# ----------------------------------------------------
		# When the stepper ID list is changed, update list.
		# ----------------------------------------------------
		# get new stepper list
		self.theStepperIDList = self.theSession.theSimulator.getStepperList()

		# when this window is shown, clear list
		if self.isShown == gtk.TRUE:
			self['stepper_id_list'].clear_items( 0, len(self.theStepperIDListItems) )

		self.theStepperIDListItems = []

		for aStepperID in self.theStepperIDList:
			aListItem = GtkListItem( aStepperID )
			aListItem.set_name( aStepperID )
			self.theStepperIDListItems.append( aListItem )

		if self.isShown == gtk.TRUE:

			if len(self.theStepperIDListItems) > 0:
				self['stepper_id_list'].append_items( self.theStepperIDListItems )

		self[self.__class__.__name__].show_all()

		# ----------------------------------------------------
		# select first item.
		# ----------------------------------------------------
		if self.isShown == gtk.TRUE:
			if len(self['stepper_id_list'].get_selection()) == 0:
				if len(self.theStepperIDListItems) >= 1:
					self['stepper_id_list'].select_item( 0 ) 
			

	# end of update


	# ---------------------------------------------------------------
	# Closes this window
	# return -> None
	# ---------------------------------------------------------------
	def closeWindow ( self, obj ):

		self[self.__class__.__name__].hide_all()

	# end of closeWindow


			

# end of StepperWindow



# ---------------------------------------------------------------
# Test code
# ---------------------------------------------------------------


if __name__ == "__main__":

	class Session:
		def __init__( self ):
			self.theSimulator = simulator()
		def getLoggerList( self ):
			#fpnlist = ((SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity'),
			#	(SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity'),
			#	(SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity'))
			fpnlist = ('Substance:/CELL/CYTOPLASM/aa:E:Quality',
					   'Substance:/CELL/CYTOPLASM/bb:F:Quality',
					   'Substance:/CELL/CYTOPLASM/cc:G:Quality')
			return fpnlist

		def getLogger( self, fpn ):
			logger = Logger( fpn )
			return logger

	class MainWindow:
		def __init__( self ):
			self.theSimulator = simulator()
			self.theRunningFlag  =0
			#theRunningFlag:
			#if self.theMainWindow.theRunningFlag:

	class simulator:

		def __init__( self ):
			self.dic={('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

		def getProperty( self, fpn ):
			return self.dic[fpn]

		def setProperty( self, fpn, value ):
			self.dic[fpn] = value

		def getLogger( self, fpn ):
			logger = Logger( fpn )
			return logger

		#def getLoggerList( self ):
		#	fpnlist = ((SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity'),
		#		(SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity'),
		#		(SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity'))
		#	return fpnlist

	class Logger:

		def __init__( self, fpn ):
			pass

		def getStartTime( self ):
			return 2

		def getEndTime( self ):
			return 108

		def getLoggerData( self ,start=0,end=0,interval=0):
			return array([[0,0],[0.1,0.1],[0.2,0.3],[0.3,0.7],[0.4,0.9],[0.5,1.0]])

		def getData( self ,start=0,end=0,interval=0):
			return array([[0,0],[0.1,0.1],[0.2,0.3],[0.3,0.7],[0.4,0.9],[0.5,1.0]])
		
              
	def mainQuit( obj, data ):
		gtk.mainquit()
		quit()
        
	def mainLoop():
		# FIXME: should be a custom function

		gtk.mainloop()

	def main():
		aMainWindow = MainWindow()
		aSession = Session()
		aLoggerWindow = LoggerWindow( aSession, aMainWindow )
		mainLoop()

	main()
