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

from OsogoWindow import *
import gobject

from os import *

from ecell.ecssupport import *
import ConfirmWindow

# Constant value of this class
PROPERTY_INDEX = 0
VALUE_INDEX = 1
GET_INDEX = 2
SET_INDEX = 3

MAX_STRING_NUMBER = 20

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
		self.theSelectedRowOfPropertyList = None
		self.theSelectedStepperIDListItem = None

	# end of the __init__


	def openWindow(self):

		OsogoWindow.openWindow(self)
		self.theStepperIDListWidget = self[ 'stepper_id_list' ]
		aListStore = gtk.ListStore( gobject.TYPE_STRING )
		self.theStepperIDListWidget.set_model( aListStore )
		column=gtk.TreeViewColumn('Stepper',gtk.CellRendererText(),text=0)
		self.theStepperIDListWidget.append_column(column)

		self.thePropertyList=self['property_list']
		aPropertyModel=gtk.ListStore( gobject.TYPE_STRING,
					 gobject.TYPE_STRING,
					  gobject.TYPE_STRING,
					   gobject.TYPE_STRING,
					    gobject.TYPE_STRING)
		self.thePropertyList.set_model(aPropertyModel)
		column=gtk.TreeViewColumn( 'Property',gtk.CellRendererText(),\
					   text=PROPERTY_INDEX )
		self.thePropertyList.append_column(column)
		column=gtk.TreeViewColumn( 'Value',gtk.CellRendererText(),\
					   text=VALUE_INDEX )
		self.thePropertyList.append_column(column)
		column=gtk.TreeViewColumn( 'Get',gtk.CellRendererText(),\
					   text=GET_INDEX )
		self.thePropertyList.append_column(column)
		column=gtk.TreeViewColumn( 'Set',gtk.CellRendererText(),\
					   text=SET_INDEX )
		self.thePropertyList.append_column(column)
		
		self.initialize()

		# -----------------------------------------------
		# Adds handlers
		# -----------------------------------------------

		self.addHandlers({ \

				# StepperID list
				'on_stepper_id_list_select'
				: self.selectStepperID, 

				# Property list
				'on_property_list_select_row'
				: self.selectProperty, 

				# update button
				'on_update_button_clicked'
				: self.updateProperty, 

				# close button
	  			'on_close_button_clicked'
				: self.closeWindow,
			})


	# end of openWindow


	# -----------------------------------------------
	# initializer
	# return -> None
	# -----------------------------------------------
	def initialize( self ):
		self.update()

	# end of initialize

	# -----------------------------------------------
	# selectStepperID
	# objects : ( stepper_id_list[List], selected_item[Item] )
	#
	# return -> None
	# -----------------------------------------------
	def selectStepperID( self, *objects ):

		# If Window is closed, do nothing.
		if self.isShown == gtk.FALSE:
			return None

		# --------------------------------------------------
		# Creates selected StepperSub 
		# --------------------------------------------------
		iter = self.theStepperIDListWidget.get_selection().get_selected()[1]
		# aStepperID is selected stepper id
		aStepperID = self.theStepperIDListWidget.get_model().get_value(iter,0)
		# aStepperStub is selected StepperStub of selected stepper
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		PropertyModel=self['property_list'].get_model()
		PropertyModel.clear()

		# aClassName = self.theSession.theSimulator.getStepperClassName( aStepperID )
		aClassName = aStepperStub.getClassname( )

		# -----------------------
		# Creates list [aList] to display
		# ---------------------------------------

		# Sets ClassName row
		aList = [ 'ClassName', ]
		aList.append( str(aClassName) )
		aList.append( decodeAttribute( TRUE ) )
		aList.append( decodeAttribute( FALSE ) )
		iter = PropertyModel.append( )
		for i in range(0,4):
	    	    PropertyModel.set_value(iter,i,aList[i])
					    
		# Sets all propertys other than ClassName

		#for aProperty in self.theSession.theSimulator.getStepperPropertyList( aStepperID ):
		for aProperty in aStepperStub.getPropertyList():

			aValue =  aStepperStub.getProperty( aProperty )

			aList = [ aProperty, ]
			aList.append( shortenString( str(aValue), MAX_STRING_NUMBER) )

			anAttribute = aStepperStub.getPropertyAttributes( aProperty )

			aList.append( decodeAttribute(anAttribute[GETABLE]) )
			aList.append( decodeAttribute(anAttribute[SETTABLE]) )
			iter = PropertyModel.append( )
			for i in range(0,4):
				PropertyModel.set_value(iter,i,aList[i])


	# -----------------------------------------------
	# selectProprety
	# *objects : the information of selected row (tuple)
	#
	# return -> None
	# -----------------------------------------------
	def selectProperty( self, *objects ):

		# If Window is closed, do nothing.
		if self.isShown == gtk.FALSE:
			return None

		# --------------------------------------------------
		# Creates selected StepperSub 
		# --------------------------------------------------
		iter = self.theStepperIDListWidget.get_selection().get_selected()[1]
		aStepperID = self.theStepperIDListWidget.get_model().get_value(iter,0)
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		# --------------------------------------------------
		# gets the number of selected row that is required in updateProprety
		# method.
		# --------------------------------------------------
		self.theSelectedRowOfPropertyList = self.thePropertyList.get_selection().get_selected()[1]

		# --------------------------------------------------
		# gets value from proprety_list
		# --------------------------------------------------
#		aValue = self['property_list'].get_model().get_value(\
#			     self.theSelectedRowOfPropertyList, VALUE_INDEX )

		aPropertyName = self['property_list'].get_model().get_value(\
			     self.theSelectedRowOfPropertyList, PROPERTY_INDEX )

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
	

		#aValue = self.theSession.theSimulator.getStepperProperty(\
		#	self.theSelectedStepperIDListItem,\
		#	aPropertyName )

		self['value_entry'].set_text( str( aValue ) )

		# --------------------------------------------------
		# when the selected property is settable, set sensitive value_entry
		# when not, set unsensitive value_entry
		# --------------------------------------------------
		if self['property_list'].get_model().get_value(  self.theSelectedRowOfPropertyList,\
								 SET_INDEX ) == decodeAttribute(TRUE):
			self['value_entry'].set_sensitive( gtk.TRUE )
			self['update_button'].set_sensitive( gtk.TRUE )
		else:
			self['value_entry'].set_sensitive( gtk.FALSE )
			self['update_button'].set_sensitive( gtk.FALSE )


	# end of selectProprety


	# -----------------------------------------------
	# updateProprety
	# This method doesn't deal with 'ClassName' is selected,
	# but there is no case that this is called with selecting 'ClassName'.
	#
	# return -> None
	# -----------------------------------------------
	def updateProperty( self, *objects ):

		# If Window is closed, do nothing.
		if self.isShown == gtk.FALSE:
			return None

		# --------------------------------------------------
		# creates selected StepperSub 
		# --------------------------------------------------
		iter = self.theStepperIDListWidget.get_selection().get_selected()[1]
		aStepperID = self.theStepperIDListWidget.get_model().get_value(iter,0)
		aStepperStub = StepperStub( self.theSession.theSimulator, aStepperID )

		# --------------------------------------------------
		# gets selected property row
		# --------------------------------------------------
		self.theSelectedRowOfPropertyList = self.thePropertyList.get_selection().get_selected()[1]
		if self.theSelectedRowOfPropertyList == None:
			aMessage = 'Select a property.'
			aDialog = ConfirmWindow.ConfirmWindow(0,aMessage,'Error!')
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
		aPropertyName = self['property_list'].get_model().get_value( self.theSelectedRowOfPropertyList,
		                                                PROPERTY_INDEX )

		# ------------------------------------
		# When the property value is scalar
		# ------------------------------------
		if type(aValue) != list and type(aValue) != tuple:

			# ---------------------------------------------------
			# converts value type
			# ---------------------------------------------------
			anOldValue = aStepperStub.getProperty( aPropertyName )

			# ---------------------------------------------------
			# sets new value
			# ---------------------------------------------------
			try:
				aStepperStub.setProperty( aPropertyName, aValue )
			except:

				import sys
				import traceback
				anErrorMessage =  string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.message( anErrorMessage )

				anErrorMessage = "Error : refer to the MessageWindow."
				aDialog = ConfirmWindow.ConfirmWindow(0,aMessage,"Can't set property!\n" + anErrorMessage)
				self['statusbar'].push(1,anErrorMessage)
				return None

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
				#self.theSession.theSimulator.setStepperProperty( self.theSelectedStepperIDListItem,
			   	#                                              aPropertyName,
			   	#                                              aNewValue )
				aStepperStub.setProperty( aPropertyName, aNewValue )

			except:

				import sys
				import traceback
#				self.printMessage(' can\'t load [%s]' %aFileName)
				anErrorMessage = \
				  string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
				self.theSession.message( aErroeMessage )

				anErrorMessage = "Error : refer to the MessageWindow."
				aDialog = ConfirmWindow.ConfirmWindow(0,aMessage,"Can't set property!\n" + anErrorMessage)
				self['statusbar'].push(1,anErrorMessage)
				return None


		# -------------------------------------------
		# updates property_list
		# -------------------------------------------
		self.selectStepperID( None)

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
		self.theStepperIDList = self.theSession.getStepperList()
		print self.theStepperIDList

                aModel = self.theStepperIDListWidget.get_model()

		# when this window is shown, clear list
		#		if self.isShown == gtk.TRUE:
		#			aModel.clear()

		aModel.clear()			
		for aValue in self.theStepperIDList:
			anIter = aModel.append()
			aModel.set( anIter, 0, aValue )

		self[self.__class__.__name__].show_all()

	# end of update

	# ---------------------------------------------------------------
	# Closes this window
	# return -> None
	# ---------------------------------------------------------------
	def closeWindow ( self, obj ):

		self[self.__class__.__name__].hide_all()
		self.isShown = FALSE
		self.theMainWindow.toggleStepperWindow()

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
			#fpnlist = ((VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value'),
			#	(VARIABLE, '/CELL/CYTOPLASM', 'ADP', 'Value'),
			#	(VARIABLE, '/CELL/CYTOPLASM', 'AMP', 'Value'))
			fpnlist = ('Variable:/CELL/CYTOPLASM/aa:E:Quality',
					   'Variable:/CELL/CYTOPLASM/bb:F:Quality',
					   'Variable:/CELL/CYTOPLASM/cc:G:Quality')
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
			self.dic={('Variable', '/CELL/CYTOPLASM', 'ATP','Value') : (1950,),}

		def getProperty( self, fpn ):
			return self.dic[fpn]

		def setProperty( self, fpn, value ):
			self.dic[fpn] = value

		def getLogger( self, fpn ):
			logger = Logger( fpn )
			return logger

		#def getLoggerList( self ):
		#	fpnlist = ((VARIABLE, '/CELL/CYTOPLASM', 'ATP', 'Value'),
		#		(VARIABLE, '/CELL/CYTOPLASM', 'ADP', 'Value'),
		#		(VARIABLE, '/CELL/CYTOPLASM', 'AMP', 'Value'))
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
