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

#from Window import *
from OsogoWindow import *
from gtk import *
from Numeric import *

from os import *

from ecell.ecssupport import *

from ecell.DataFileManager import *
from ecell.ECDDataFile import *

# ---------------------------------------------------------------
# This is LoggerWindow class .
#
# ---------------------------------------------------------------
class LoggerWindow(OsogoWindow):


	# ---------------------------------------------------------------
	# constructor
	# aSession : the reference of session
	# aMainWindow : the reference of MainWindow
	# ---------------------------------------------------------------
	def __init__( self, aSession, aMainWindow ): 

		#OsogoWindow.__init__( self, 'LoggerWindow.glade', aMainWindow )
		OsogoWindow.__init__( self, aMainWindow, 'LoggerWindow.glade' )
		self.theSession = aSession
		self.theDefaultSaveDirectory='Data'
		self.thePopupMenu = PopupMenu( self )
		self.theList = []

		# ---------------------------------------------------------------
		# Creates save file selection
		# ---------------------------------------------------------------
		self.theSaveDirectorySelection = gtk.GtkFileSelection( 'Select Rule File' )
		self.theSaveDirectorySelection.ok_button.connect('clicked', self.changeSaveDirectory)
		self.theSaveDirectorySelection.cancel_button.connect('clicked', self.closeParentWindow)

	# end of the __init__


	def openWindow(self):

		OsogoWindow.openWindow(self)
		#self.theMainWindow = aMainWindow

		self.theEntryList = self['loggerWindow_clist']
		self.initialize()

		# reset
		self.resetAllValues()

		# ---------------------------------------------------------------
		# Adds handlers
		# ---------------------------------------------------------------

		self.addHandlers({ \

				# bottom basic bottons (save,close,reset)
				'on_save_button_clicked' : self.saveData, 
				'on_close_button_clicked' : self.closeWindow, 
				'on_reset_button_clicked' : self.resetAllValues, 

				# save_directory 
	  			'on_directory_button_clicked' : self.spawnSaveDirectorySelection,

				# data_interval
	  			'on_datainterval_checkbox_toggled' : self.updateDataInterval,

				# time
	  			'on_time_checkbox_toggled' : self.updateStartEnd,

				# window event
		 		'on_exit_activate' : self.closeWindow,

				# popup
				'button_press_event'  : self.popupMenu,

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
	# Resets all value
	# return -> None
	# ---------------------------------------------------------------
	def resetAllValues( self, obj=None ):

		# data_type
		pass

		# save_directory
		self['directory_entry'].set_text(self.theDefaultSaveDirectory)

		# data_interval
		self['datainterval_checkbox'].set_active(1)
		self['datainterval_spinbutton'].set_sensitive(0)

		# specity_the_time_to_save
		self['time_checkbox'].set_active(1)
		self['start_spinbutton'].set_sensitive(0)
		self['end_spinbutton'].set_sensitive(0)

		self.update()

	# end of resetAllValues


	# ---------------------------------------------------------------
	# Closes parent window
	# return -> None
	# ---------------------------------------------------------------
	def closeParentWindow( self, obj ):
		aParentWindow = self.theSaveDirectorySelection.cancel_button.get_parent_window()
		aParentWindow.hide()

	# end of closeParentWindow


	# ---------------------------------------------------------------
	# Spawns
	# return -> None
	# ---------------------------------------------------------------
	def spawnSaveDirectorySelection( self, obj ):
		self.theSaveDirectorySelection.show_all()

	# end of spawnSaveDirectorySelection


	# ---------------------------------------------------------------
	# If directory_button is clicked, this method is called.
	# return -> None
	# ---------------------------------------------------------------
	def changeSaveDirectory( self, obj ):
		aSaveDirectory = self.theSaveDirectorySelection.get_filename()
		self['directory_entry'].set_text(aSaveDirectory)
		self.theSaveDirectorySelection.hide()

	# end of changeSaveDirectory


	# ---------------------------------------------------------------
	# If datainterval_checkbox is toggled, this method is called.
	# return -> None
	# ---------------------------------------------------------------
	def updateDataInterval( self, obj=None ):
		if self['datainterval_checkbox'].get_active():
			self['datainterval_spinbutton'].set_sensitive(0)
		else:
			self['datainterval_spinbutton'].set_sensitive(1)

	# end of updateDataInterval


	# ---------------------------------------------------------------
	# If time_checkbox is toggled, this method is called.
	# return -> None
	# ---------------------------------------------------------------
	def updateStartEnd( self, obj=None ):
		if self['time_checkbox'].get_active():
			self['start_spinbutton'].set_sensitive(0)
			self['end_spinbutton'].set_sensitive(0)
		else:
			self['start_spinbutton'].set_sensitive(1)
			self['end_spinbutton'].set_sensitive(1)

	# end of updateStartEnd


	# ---------------------------------------------------------------
	# If save_button is clicked, then this method is called.
	# return -> None
	# ---------------------------------------------------------------
	def saveData( self, obj ):

		self.aGeneralSaveErrorMessage="couldn't save files."

		# -------------------------------------------------
		# [0] clean up statusbar
		# -------------------------------------------------
		self["statusbar"].pop(1)

		# -------------------------------------------------
		# [1] checks status and value of each widget
		# -------------------------------------------------

		# [1-1] At least, one data must be selected.
		# If no list is selected, exit this method.
		if len(self.theSelectedPropertyName())==0:
			self["statusbar"].push(1,'Select some data.')
			aErrorMessage='\nNo data is selected.!\n'
			aWarningWindow = ConfirmWindow(0,aErrorMessage)
			return None

		# [1-2] interval must be > 0
		# If interval is 0, exit this method.
		aInterval = -1
		if not self['datainterval_checkbox'].get_active():
			aInterval = self['datainterval_spinbutton'].get_value()

			if aInterval==0:
				self["statusbar"].push(1,'Set interval > 0.')
				aErrorMessage='Interval must be > 0.!\n'
				aWarningWindow = ConfirmWindow(0,aErrorMessage)
				return None


		# [1-3] Now binary type is not supported by Logger.
		# If binary type is selected, exit this method.
		aType = self["datatype_combo"].get_text()

		if aType == 'ecd':
			pass
		elif aType == 'binary':
			self["statusbar"].push(1,'Select ecd type.')
			aErrorMessage = "Sorry, binary format will be supported in the future version."
			aWarningWindow = ConfirmWindow(0,aErrorMessage)
			return None

		# [1-4] start < end
		# If start >= end, exit this method.
		aStartTime=-1
		aEndTime=-1
		if not self['time_checkbox'].get_active():
			aStartTime = self['start_spinbutton'].get_value()
			aEndTime = self['end_spinbutton'].get_value()

			if aStartTime >= aEndTime:
				self["statusbar"].push(1,'Set start time < end time.')
				aErrorMessage='Start time must be < end time.!\n'
				aWarningWindow = ConfirmWindow(0,aErrorMessage)
				return None

		# -------------------------------------------------
		# [2] Creates Data directory.
		# -------------------------------------------------
		aSaveDirectory = self['directory_entry'].get_text()

		# If same directory exists.
		if os.path.isdir(aSaveDirectory):
			aConfirmMessage = "%s directory already exist.\n Would you like to override it?"%aSaveDirectory
			self.confirmWindow = ConfirmWindow(1,aConfirmMessage)

			if self.confirmWindow.return_result() == 0:
				pass
			else:
				self["statusbar"].push(1,'Save was canceled.')
				return None

		# If same directory dose not exists.
		else:
			try:
				os.mkdir(aSaveDirectory)
				self["statusbar"].push(1,'Set start time < end time.')
			except:
				self["statusbar"].push(1,'couldn\'t create %s'%aSaveDirectory)
				aErrorMessage='couldn\'t create %s!\n'%aSaveDirectory
				aWarningWindow = ConfirmWindow(0,aErrorMessage)
				return None
			else:
				self["statusbar"].push(1,'%s was created.'%aSaveDirectory)


		# -------------------------------------------------
		# [3] Execute saving.
		# -------------------------------------------------
		#aLoggerFactory = LoggerFactory()

		# creates instance datafilemanager
		aDataFileManager = DataFileManager()

		# sets root directory to datafilemanager
		aDataFileManager.setRootDirectory(aSaveDirectory)

		aFileIndex=0

		try: #(1)

			# gets all list of selected property name
			for aFullPNString in self.theSelectedPropertyName(): #(2)

				# -------------------------------------------------
				# creates filename
				# from [Substance:/CELL/CYTOPLASM:E:Quantity] 
				# to   [CYTOPLASM-E-Quantity]
				# -------------------------------------------------

				aFileName=split(join(split(aFullPNString,':')[1:],'-'),'/')[-1]

				aECDDataFile = ECDDataFile()
				aECDDataFile.setFileName(aFileName)

				# -------------------------------------------------
				# Gets logger
				# -------------------------------------------------
				aLogger = self.theSession.theSimulator.getLogger( aFullPNString )
				if aStartTime == -1 or aEndTime == -1:
					# gets start time and end time from logger
					aStartTime= aLogger.getStartTime()
					aEndTime  = aLogger.getEndTime()
				else:
					# checks the value
					if not (aLogger.getStartTime() < aStartTime < aLogger.getEndTime()):
						aStartTime = aLogger.getStartTime()
					if not (aLogger.getStartTime() < aEndTime < aLogger.getEndTime()):
						aEndTime = aLogger.getEndTime()

				# -------------------------------------------------
				# gets the matrix data from logger.
				# -------------------------------------------------
				if aInterval == -1:
					# gets data with specifing interval 
					aMatrixData = aLogger.getData(aStartTime,aEndTime)
				else:
					# gets data without specifing interval 
					aMatrixData = aLogger.getData(aStartTime,aEndTime,aInterval)


				# sets data name 
				aECDDataFile.setDataName(aFullPNString)

				# sets matrix data
				aECDDataFile.setMatrixData(aMatrixData)

				# -------------------------------------------------
				# adds data file to data file manager
				# -------------------------------------------------
				aDataFileManager.getFileMap()[`aFileIndex`] = aECDDataFile

				aFileIndex = aFileIndex + 1

			# for(2)

			aDataFileManager.saveAll()

		except: #try(1)

			# -------------------------------------------------
			# displays error message and exit this method.
			# -------------------------------------------------

			import sys
			print __name__,
			print sys.exc_traceback
			aErrorMessage= "Error : could not save [%s] " %aFullPNString
			self["statusbar"].push(1,aErrorMessage)
			return None

		else: # try(1)
			# -------------------------------------------------
			# displays error message and exit this method.
			# -------------------------------------------------

			aSuccessMessage= " All files you selected are saved. " 
			self["statusbar"].push(1,aSuccessMessage)

		# end of try(1)

	# end of saveData


	# ---------------------------------------------------------------
	# Gets the selected PropertyName
	# return -> selected propertyname list
	# ---------------------------------------------------------------
	def theSelectedPropertyName( self ):
		aSelectedPropertyNameList=[]
		for aRowNumber in self["loggerWindow_clist"].selection:
			aPropertyName = self["loggerWindow_clist"].get_text(aRowNumber,0)
			aSelectedPropertyNameList.append(aPropertyName)
		return aSelectedPropertyNameList

	# end of theSelectedPropertyName


	# ---------------------------------------------------------------
	# Selects PropertyName
	# return -> None
	# ---------------------------------------------------------------
	def selectPropertyName( self ):
		aCList = self['loggerWindow_clist']

		for aRowNumber in aCList.selection:
			aPropertyName = aCList.get_text(aRowNumber,0)

	# end of selectPropertyName


	# ---------------------------------------------------------------
	# Updates
	# return -> None
	# ---------------------------------------------------------------
	def update( self ):

		if self.isShown == gtk.TRUE:
			self.theEntryList.clear()

		self.theFullPNList = self.theSession.getLoggerList()
		self.theList = []

		for aFullPNString in self.theFullPNList :

			aLogger = self.theSession.getLogger( aFullPNString )
			start = str( aLogger.getStartTime() )
			if self.theMainWindow.theRunningFlag:
				end = 'running'
			else:
				end = str( aLogger.getEndTime() )
			aList = [ aFullPNString, start, end ]
			self.theList.append( aList )

		for aValue in self.theList:
			self.theEntryList.append( aValue )
	
	# update


	# ---------------------------------------------------------------
	# Closes this window
	# return -> None
	# ---------------------------------------------------------------
	def closeWindow ( self, obj ):
		#gtk.mainquit()
		self[self.__class__.__name__].hide_all()

	# closeWindow

	# ---------------------------------------------------------------
	# popupMenu
	#   - show popup menu
	#
	# aWidget         : widget
	# anEvent          : an event
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def popupMenu( self, aWidget, anEvent ):

		if len(self.theSelectedPropertyName())!=0:
			if anEvent.button == 3:
				self.thePopupMenu.popup( None, None, None, 1, 0 )

	# end of poppuMenu


	def deleteItem( self, anObject ):
		#print "deleteIetm -- s "
		#print anObject
		#for aSelectedFullPNString in self.theSelectedPropertyName():
		#	print aSelectedFullPNString
		#	self.theSession.getLoggerList().remove( aSelectedFullPNString )
		#print "deleteIetm -- e "
		pass
			

# end of LoggerWindow


# ---------------------------------------------------------------
# PopupMenu -> GtkMenu
# ---------------------------------------------------------------
class PopupMenu( GtkMenu ):

	# ---------------------------------------------------------------
	# Constructor
	#
	# aParent        : parent plugin window
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aParent ):

		# ------------------------------------------
		# calls the constructor of super class
		# ------------------------------------------
		GtkMenu.__init__(self)

		aMaxStringLength = 0
		aMenuSize = 0

		# ------------------------------------------
		# sets arguments to instance valiables
		# ------------------------------------------
		self.theParent = aParent

		# ------------------------------------------
		# initializes menu item
		# ------------------------------------------
		self.theMenuItem = {}

		self.theDeleteString = 'delete this item'
		self.theMenuItem[self.theDeleteString]= GtkMenuItem(self.theDeleteString)
		self.theMenuItem[self.theDeleteString].connect('activate', self.theParent.deleteItem )
		self.theMenuItem[self.theDeleteString].set_name(self.theDeleteString)
		self.append( self.theMenuItem[self.theDeleteString] )

		self.theMenuItem[self.theDeleteString].set_sensitive(0)

		aMaxStringLength = len(self.theDeleteString)

		# ------------------------------------------
		# caliculates size of menu and sets it to itself
		# ------------------------------------------
		self.theWidth = (aMaxStringLength+1)*8
		self.theHeight = (aMenuSize+1)*21 + 3
		self.set_usize( self.theWidth, self.theHeight )

	def popup(self, pms, pmi, func, button, time):
		GtkMenu.popup(self, pms, pmi, func, button, time)
		self.show_all(self)


# end of PopupMenu


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

		def getEntityProperty( self, fpn ):
			return self.dic[fpn]

		def setEntityProperty( self, fpn, value ):
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
