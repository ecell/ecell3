#! /usr/bin/env python
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
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
# 	      'Masahiro Sugimoto <sugi@bioinformatics.org>' 
# 	      'Gabor Bereczki <gabor.bereczki@talk21.com>' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from gui_config import *
import sys

sys.path.append(GUI_OSOGO_PATH)
import MainWindow  
from ecell.Session import *
import gtk

import EntityListWindow
import LoggerWindow
import InterfaceWindow 
import StepperWindow 
import BoardWindow 
import ConfigParser

from ConfirmWindow import *

from OsogoPluginManager import *

class GtkSessionMonitor(Session):
#
#	GtkSessionMonitor class functions
#

	# ==========================================================================
	def __init__(self, aSimulator = None ):
		"""sets up the osogo session, creates Mainwindow and other fundamental
		windows but doesn't show them"""

		#calls superclass
	
		Session.__init__(self, aSimulator )

		# -------------------------------------
		# reads defaults from osogo.ini 
		# -------------------------------------
		self.theConfigDB=ConfigParser.ConfigParser()
    		self.theConfigDB.read(OSOGO_PATH+os.sep+'osogo.ini')

		self.theUpdateInterval = 150
		self.theStepSizeOrSec = 1.0
		self.theRunningFlag = 0

		# -------------------------------------
		# creates PluginManager
		# -------------------------------------
		self.thePluginManager = OsogoPluginManager( self )
		self.thePluginManager.loadAll()

		# -------------------------------------
		# creates FundamentalWindow
		# -------------------------------------
		
		# key:window name(str) value:window instance
		self.theFundamentalWindows = {}

		# creates fundamental windows
		aLoggerWindow     = LoggerWindow.LoggerWindow(  self )
		anInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
		aStepperWindow    = StepperWindow.StepperWindow(  self )
		aBoardWindow      = BoardWindow.BoardWindow(  self )
		aMainWindow	  = MainWindow.MainWindow( self ) 

		# saves them to map
		self.theFundamentalWindows['LoggerWindow'] = aLoggerWindow
		self.theFundamentalWindows['InterfaceWindow'] = anInterfaceWindow
		self.theFundamentalWindows['StepperWindow'] = aStepperWindow
		self.theFundamentalWindows['BoardWindow'] = aBoardWindow
		self.theFundamentalWindows['MainWindow'] = aMainWindow

		# key:EntityListWindow instance value:None
		# In deleteEntityListWindow method, an instance of EntityListWindow is
		# accessed directory. The sequence information of EntityListWindow does
		# not need. So the references to EntityListWindow instances should be 
		# held dict's key. Values of dict are not also imported.

		self.theEntityListInstanceMap = {}  

		# -------------------------------------
		# initializes for run method 
		# -------------------------------------
		self.theSimulator.setEventChecker( gtk.events_pending )
		self.theSimulator.setEventHandler( gtk.mainiteration  )		

		# -------------------------------------
		# creates MainWindow
		# -------------------------------------
	
		self.theMainWindow = aMainWindow
	# ==========================================================================
	def synchronize(self):
		self.updateWindows()
		while gtk.events_pending():
			gtk.main_iteration_do()

	# ==========================================================================
	def GUI_interact(self):			#
		"hands over controlto the user (gtk.mainloop())"

		gtk.mainloop()

	# ==========================================================================
	def QuitGUI( self ):
		""" quits gtk.mainloop() after saving changes """
		gtk.mainquit()

	# ==========================================================================
	def doesExist( self, aWindowName):
		""" aWindowName: (str) name of Window
		     returns gtk.TRUE if window is opened
			     gtk.FALSE if window is not opened
		     checks both plugin and fundamental windows 
		"""

		# check fundamentalwindows
		if self.theFundamentalWindows.has_key(aWindowName):
			return self.theFundamentalWindows[ aWindowName ].exists()

		# check entity list windows
		if aWindowName == 'EntityListWindow' and len( self.theEntityListInstanceMap>0):
			return gtk.TRUE
		# check pluginwindow instances
		
		aPluginInstanceList = self.thePluginManager.thePluginTitleDict.keys()

		for aPluginInstance in aPluginInstanceList:
			if aWindowName == self.thePluginManager.thePluginTitleDict[aPluginInstance]:
				return gtk.TRUE
		return gtk.FALSE


	# ==========================================================================
	def openWindow( self, aWindowName ): 		#
		"""opens up window and returns aWindowname instance
		aWindowName   ---  Window name (str)
		Returns FundamentalWindow or EntityListWindow list
		"""
		if len(self.theModelName) == 0 and aWindowName != 'MainWindow':
			message ( "Model has not yet been loaded. Can't open windows." )
			return None
		# When the WindowName does not matched, creates nothing.
		if self.theFundamentalWindows.has_key( aWindowName ) == TRUE:
			self.theFundamentalWindows[ aWindowName ].openWindow()
			self.theMainWindow.updateButtons()
			return self.theFundamentalWindows[ aWindowName ]
		elif aWindowName == 'EntityListWindow':
			return self.createEntityListWindow()
		else:
			message( "No such WindowType (%s) " %aWindowName )
			return None

	# ==========================================================================
	def getWindow( self, aWindowName ):
		"""
		aWindowName   ---  Window name (str)
		Returns FundamentalWindow or EntityListWindow list
		"""

		# check fundamentalwindows
		if self.theFundamentalWindows.has_key(aWindowName):
			return self.theFundamentalWindows[aWindowName]

		# check entity list windows
		if aWindowName == 'EntityListWindow':
			return self.theEntityListInstanceMap.keys()
		# check pluginwindow instances
		
		aPluginInstanceList = self.thePluginManager.thePluginTitleDict.keys()

		for aPluginInstance in aPluginInstanceList:
			aWindowName = self.thePluginManager.thePluginTitleDict[aPluginInstance]
			return aPluginInstance
		return None


	# ==========================================================================
	def displayWindow( self, aWindowName ):
		"""When the Window is not created, calls its openWidow() method.
		When already created, move it to the top of desktop.
		aWindowName   ---  window name(str)
		Return None
		[None]:When the WindowName does not matched, creates nothing.
		"""

		# When the WindowName does not match, creates nothing.
		if self.theFundamentalWindows.has_key( aWindowName ) == FALSE:
			message ( "No such WindowType (%s) " %aWindowName )
			return None

		# When the Window is already created, move it to the top of desktop
		if self.theFundamentalWindows[aWindowName].exists():
			self.theFundamentalWindows[aWindowName].present()
			pass
		else:
			self.theFundamentalWindows[aWindowName].openWindow()
			self.theFundamentalWindows[aWindowName].update()
	
	# ==========================================================================
	def toggleWindow( self, aWindowName ):
		if self.theFundamentalWindows[aWindowName].exists():
			self.theFundamentalWindows[aWindowName].close()
			
		else:
			self.theFundamentalWindows[aWindowName].openWindow()
			self.theFundamentalWindows[aWindowName].update()
		if self.theFundamentalWindows['MainWindow'].exists():
			self.theFundamentalWindows['MainWindow'].update()


	# ==========================================================================
	def createPluginWindow(self, aType, aFullPNList):
		""" opens and returns _PluginWindow instance of aType showing aFullPNList 
			returns None if pluginwindow could not have been created """
		anInstance = self.thePluginManager.createInstance( aType, aFullPNList)
		if anInstance == None:
			self.message ( 'Pluginwindow has not been created. %s may not be a valid plugin type' %aType )
		return anInstance


	# ==========================================================================
	def createPluginOnBoard(self, aType, aFullPNList):	
		""" creates and adds plugin to pluginwindow and returns plugininstance """
		aBoardWindow = self.getWindow('BoardWindow')
		if aBoardWindow == None:
			self.message('Board Window does not exist. Plugin cannot be added.')
			return None
		return aBoardWindow.addPluginWindows( aType, aFullPNList)


	# ==========================================================================
	def openConfirmWindow(self,  aMessage, aTitle ):
		""" pops up a modal dialog window
			with aTitle (str) as its title
			and displaying aMessage as its message
			and with an OK and a Cancel button
			returns:
			True if Ok button is pressed
			False if cancel button is pressed
		"""
		aConfirmWindow = ConfirmWindow(1, aMessage, aTitle )
		return aConfirmWindow.return_result() == OK_PRESSED


	# ==========================================================================
	def createEntityListWindow( self ):
		"""creates and returns an EntityListWindow
		"""
		anEntityListWindow = None
		# when Model is already loaded.
		if len(self.theModelName) > 0:

			# creates new EntityListWindow instance
			anEntityListWindow = EntityListWindow.EntityListWindow( self )
			anEntityListWindow.openWindow()

			# saves the instance into map
			self.theEntityListInstanceMap[ anEntityListWindow ] = None
		
			# updates all fundamental windows
			self.updateFundamentalWindows()
		return anEntityListWindow


	# ==========================================================================
	def deleteEntityListWindow( self, anEntityListWindow ):
		"""deletes the reference to the instance of EntityListWindow
		anEntityListWindow   ---  an instance of EntityListWindow(EntityListWindow)
		Return None
		[Note]: When the argument is not anEntityListWindow, throws exception.
		        When this has not the reference to the argument, does nothing.
		"""

		# When the argument is not anEntityListWindow, throws exception.
		if anEntityListWindow.__class__.__name__ != 'EntityListWindow':
			raise "(%s) must be EntityListWindow" %anEntityListWindow

		# deletes the reference to the PropertyWindow instance on the EntityListWindow
		self.thePluginManager.deletePropertyWindowOnEntityListWinsow( anEntityListWindow.thePropertyWindow )

		# deletes the reference to the EntityListWindow instance
		if self.theEntityListInstanceMap.has_key( anEntityListWindow ):
			anEntityListWindow.close()
			del self.theEntityListInstanceMap[ anEntityListWindow ]
	
	# ==========================================================================
	def __updateByTimeOut( self, arg ):
		"""when time out, calls updates method()
		Returns None
		"""
		self.updateWindows()
		self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.__updateByTimeOut, 0 )


	# ==========================================================================
	def __removeTimeOut( self ):
		"""removes time out
		Returns None
		"""

		gtk.timeout_remove( self.theTimer )

	# ==========================================================================
	def updateWindows( self ):
		self.theMainWindow.update()
		self.updateFundamentalWindows()
		# updates all plugin windows
		self.thePluginManager.updateAllPluginWindow()
 	

	# ==========================================================================
	def setUpdateInterval(self, Secs):		#
		"plugins are refreshed every secs seconds"
		self.theMainWindow.theUpdateInterval = Secs
	
	# ==========================================================================
	def getUpdateInterval(self ):		#
		"returns the rate by plugins are refreshed "
		return self.theMainWindow.theUpdateInterval 


	# ==========================================================================
	def updateFundamentalWindows( self ):
		"""updates fundamental windows
		Return None
		"""

		# updates all fundamental windows
		for aFundamentalWindow in self.theFundamentalWindows.values():
			aFundamentalWindow.update()

		# updates all EntityListWindow
		for anEntityListWindow in self.theEntityListInstanceMap.keys():
			anEntityListWindow.update()

		#update MainWindow
		self.theMainWindow.update()


	# ==========================================================================
	def __readIni(self,aPath):
		"""read osogo.ini file
		an osogo.ini file may be in the given path
		that have an osogo section or others but no default
		argument may be a filename as well
		"""

	    # first delete every section apart from default
		for aSection in self.theConfigDB.sections():
			self.theConfigDB.remove(aSection)

		# gets pathname
		if not os.path.isdir( aPath ):
			aPath=os.path.dirname( aPath )

		# checks whether file exists
		aFilename=aPath+os.sep+'osogo.ini'
		if not os.path.isfile( aFilename ):
			# self.message('There is no osogo.ini file in this directory.\n Falling back to system defauls.\n')
			return None

	    # tries to read file

		try:
			self.message('Reading osogo.ini file from directory [%s]' %aPath)
			self.theConfigDB.read( aFilename )

		# catch exceptions
		except:
			self.message(' error while executing ini file [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.message(anErrorMessage)

	# ==========================================================================
	def getParameter(self, aParameter):
		"""tries to get a parameter from ConfigDB
		if the param is not present in either osogo or default section
		raises exception and quits
		"""

		# first try to get it from osogo section
		if self.theConfigDB.has_section('osogo'):
			if self.theConfigDB.has_option('osogo',aParameter):
				return self.theConfigDB.get('osogo',aParameter)

		# gets it from default
		return self.theConfigDB.get('DEFAULT',aParameter)

	# ==========================================================================
	def setParameter(self, aParameter, aValue):
		"""tries to set a parameter in ConfigDB
		if the param is not present in either osogo or default section
		raises exception and quits
		"""
		# first try to set it in osogo section
		if self.theConfigDB.has_section('osogo'):
			if self.theConfigDB.has_option('osogo',aParameter):
				self.theConfigDB.set('osogo',aParameter, str(aValue))
		else:

			# sets it in default
			self.theConfigDB.set('DEFAULT',aParameter, str(aValue))


#------------------------------------------------------------------------
#IMPORTANT!
#
#Session methods to be used in interactive scripting shoould be overloaded here
#-------------------------------------------------------------------------

	#-------------------------------------------------------------------
	def loadScript( self, ecs, parameters={} ):
		self.__readIni( ecs )
		Session.loadScript (self, ecs, parameters )

	#-------------------------------------------------------------------
	def interact( self, parameters={} ):
		Session.interact (self, parameters )

	#-------------------------------------------------------------------
	def loadModel( self, aModel ):
		self.__readIni( aModel )
		Session.loadModel( self, aModel )

	#-------------------------------------------------------------------
	def saveModel( self , aModel ):
		Session.saveModel( self , aModel )

	#-------------------------------------------------------------------
	def setMessageMethod( self, aMethod ):
		Session.setMessageMethod( self, aMethod )

	#-------------------------------------------------------------------
	def message( self, message ):
		Session.message( self, message )
		#self._synchronize()

	#-------------------------------------------------------------------
	def run( self , time = '' ):
		""" 
		if already running: do nothing
		if time is given, run for the given time
		if time is not given:
			if Mainwindow is not opened create a stop button
			set up a timeout rutin and Running Flag 
		"""

		if self.theRunningFlag == TRUE:
			return

		if time == '' and not self.doesExist('MainWindow'):
			self.openWindow('MainWindow')

		try:
			self.theRunningFlag = TRUE
			# this can fail if the simulator is not ready
			self.theSimulator.initialize()

			aCurrentTime = self.getCurrentTime()
			self.message("%15s"%aCurrentTime + ":Start\n" )
			self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.__updateByTimeOut, FALSE)
			Session.run( self, time )
			self.theRunningFlag = FALSE
			self.__removeTimeOut()

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.message(anErrorMessage)
			self.theRunningFlag = 0

		self.updateWindows()

	#-------------------------------------------------------------------
	def stop( self ):
		""" stop Simulation, remove timeout, set Running flag to false
		"""


		try:
			if self.theRunningFlag == TRUE:
				Session.stop( self )

				aCurrentTime = self.getCurrentTime()
				self.message( ("%15s"%aCurrentTime + ":Stop\n" ))
				self.__removeTimeOut()
				self.theRunningFlag = FALSE

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.message(anErrorMessage)
		self.updateWindows()
		#self._synchronize()

	#-------------------------------------------------------------------
	def step( self, num = None ):
		""" step according to num, if num is not given,
			according to set step parameters
		"""
		if self.theRunningFlag == 1:
			return

		if num == None:
			#set it to 1
			    num = 1
			    self.message( "Zero step value overridden to 1\n" )

		try:
			self.theRunningFlag = 1
			# this can fail if the simulator is not ready
			self.theSimulator.initialize()

			self.message( "Step\n" )
			self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.__updateByTimeOut, 0 )

			Session.step( self, int( num ) )

			self.theRunningFlag = 0
			self.__removeTimeOut()

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.message( anErrorMessage )
			self.theRunningFlag = 0

		self.updateWindows()
		#self._synchronize()

	def isRunning(self):
		return self.theRunningFlag

	#-------------------------------------------------------------------
	def getNextEvent( self ):
		return Session.getNextEvent( self )

	#-------------------------------------------------------------------
	def getCurrentTime( self ):
		return Session.getCurrentTime( self )

	#-------------------------------------------------------------------
	def setEventChecker( self, event ):
		Session.setEventChecker( self, event )

	#-------------------------------------------------------------------
	def setEventHandler( self, event ):
		Session.setEventHandler( self, event )

	#-------------------------------------------------------------------
	def initialize( self ):
		Session.initialize()

	#-------------------------------------------------------------------
	def getStepperList( self ):
		return Session.getStepperList( self )

	#-------------------------------------------------------------------
	def createStepperStub( self, id ):
		return Session.createStepperStub( self, id )

	#-------------------------------------------------------------------
	def getEntityList( self, entityType, systemPath ):
		return Session.getEntityList( self, entityType, systemPath )

	#-------------------------------------------------------------------
	def createEntityStub( self, fullid ):
		return Session.createEntityStub( self, fullid )

	#-------------------------------------------------------------------
	def getLoggerList( self ):
		return Session.getLoggerList( self )

	#-------------------------------------------------------------------
	def createLogger( self, fullpn ):
		Session.createLogger( self, fullpn )
#FIXME		#remember refresh Tracer and Loggerwindows!!!

	#-------------------------------------------------------------------
	def createLoggerStub( self, fullpn ):
		return Session.createLoggerStub( self, fullpn )

	#-------------------------------------------------------------------
	def saveLoggerData( self, fullpn=0, aSaveDirectory='./Data', aStartTime=-1, anEndTime=-1, anInterval=-1 ):
		Session.saveLoggerData( self, fullpn=0, aSaveDirectory='./Data', aStartTime=-1, anEndTime=-1, anInterval=-1 )



