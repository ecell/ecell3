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
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito',
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.

import os

from config import *
from gtk import *
from ecell.ecs_constants import *
import string
import sys
import traceback
from ecell.ecssupport import *

#from PluginWindow import *
from ecell.PluginWindow import *
from OsogoUtil import *
from ConfirmWindow import *

class OsogoPluginWindow(PluginWindow):
	"""OsogoPluginWindow
	This class has the following attribute and methods.

	self.theRawFullPNList : [ FullPN1, FullID2, FullPN3, , , ]
	theFullPNList()       : [ FullPN1, FullPN2, FullPN3, , , ]
	theFullIDList()       : [ FullID1, FullID2, FullID3, , , ]
	theFullPN()           : FullPN1
	theFullID()           : FullID1
	[Note]:When the Property of FullPN is wrong, the constructor of subclass
	       should throw TypeError. PluginManager will catch this Error,
	       display error message and create nothing.	
	"""

	# If the window displays multiple FullPN/FullID, theViewType is MULTIPLE
	theViewType = SINGLE  # default

	# ========================================================================
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):
		"""Constructor
		aDirName        --  a directory name including plugin module
		                    (str:absolute path/relative path)
		aData           --  a RawFullPNList (RawFullPNList)
		aPluginManager  --  a reference to PluginManager (PluginManager)
		aRoot           --  a root property (str)
		"""

		#self.theFullPNListClipBoard = []
		self.theSelectedFullPNIndex = 0

		# calls superclass's constructor
		PluginWindow.__init__( self, aDirName, aPluginManager, aRoot )

		self.theSession = self.thePluginManager.theSession 
		self.theRawFullPNList = aData

		# sets default title
		self.theTitle = self.__class__.__name__


	# ========================================================================
	def openWindow( self ):
		"""overwrites superclass's method
		Returns None
		[Note]:When this is top window, appends 'destroy' signal handler.
		"""

		# calls superclass's method
		PluginWindow.openWindow( self )

		# When this is top window, appends 'destroy' signal handler.
		if self.theRoot == None:
			self[self.__class__.__name__].connect('destroy',self.exit)


	# ========================================================================
	def setRawFullPNList( self, aRawFullPNList ):
		"""sets RawFullPNList
		aRawFullPNList  --  a RawFullPNList to be set (RawFullPNList)
		Returns None
		"""

		self.theRawFullPNList = aRawFullPNList


	# ========================================================================
	def appendRawFullPNList( self, aRawFullPNList ):
		"""appneds RawFullPNList
		aRawFullPNList  --  a RawFullPNList to be appned (RawFullPNList)
		Returns None
		"""

		self.theRawFullPNList += aRawFullPNList 

	# ---------------------------------------------------------------
	# getRawFullPNList
	#   - return RawFullPNList
	#
	# return -> RawFullPNList
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def getRawFullPNList( self ):
		return self.theRawFullPNList 

	# ---------------------------------------------------------------
	# theFullPNList
	#   - return FullPNList
	#
	# return -> FullPNList
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def theFullPNList( self ):

		return map( self.supplementFullPN, self.theRawFullPNList )

	# end of theFullPNList


	# ---------------------------------------------------------------
	# theFullIDList
	#   - return FullIDList
	#
	# return -> FullIDList
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def theFullIDList( self ):

		return map( convertFullPNToFullID, self.theRawFullPNList )

	# end of theFullIDList


	# ---------------------------------------------------------------
	# theFullPN
	#   - return FullPN
	#
	# return -> FullPN
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def theFullPN( self ):

		return self.supplementFullPN( self.theRawFullPNList[self.theSelectedFullPNIndex] )

	# end of theFullPN


	# ---------------------------------------------------------------
	# theFullID
	#   - return FullID
	#
	# return -> FullID
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def theFullID( self ):

		return convertFullPNToFullID( self.theFullPN() )

	# end of theFullID


	# ---------------------------------------------------------------
	# supplementFullID
	#   - supplements default parameter to FullID
	#   - return the supplemented FullID
	#
	# return -> supplemented FullID
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def supplementFullPN( self, aFullPN ):

		if aFullPN[PROPERTY] != '' :
			return aFullPN
		else :
			if aFullPN[TYPE] == VARIABLE :
				aPropertyName = 'Value'
			elif aFullPN[TYPE] == PROCESS :
				aPropertyName = 'Activity'
			elif aFullPN[TYPE] == SYSTEM :
				aPropertyName = 'Size'
			aNewFullPN = convertFullIDToFullPN( convertFullPNToFullID(aFullPN), aPropertyName )
			return aNewFullPN

	# end of supplementFullPN


	# ---------------------------------------------------------------
	# getValue from the session.simulator
	#   - return a value
	#
	# aFullPN : FullPN
	# return -> attribute map 
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def getValue( self, aFullPN ):

		return self.theSession.theSimulator.getEntityProperty( createFullPNString( aFullPN ) )

	# getValue


	# ---------------------------------------------------------------
	# setValue 
	#   - sets value to the session.simulator
	#
	# aFullPN : FullPN
	# aValue  : one element or tuple
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def setValue( self, aFullPN, aValue ):

		aFullID = convertFullPNToFullID( aFullPN )
		aPropertyList = self.theSession.theSimulator.getEntityPropertyList( createFullIDString( aFullID ) )
		anAttribute = self.theSession.theSimulator.getEntityPropertyAttributes( createFullPNString( aFullPN ) )

		if anAttribute[SETTABLE] == TRUE:
			self.theSession.theSimulator.setEntityProperty( createFullPNString( aFullPN ), aValue )

			self.thePluginManager.updateAllPluginWindow()
			self.thePluginManager.theSession.updateFundamentalWindows()

			#return None
		else:
			aFullPNString = createFullPNString( aFullPN )
			self.theSession.message('%s is not settable' % aFullPNString )
			#return None
	
	# end of setValue


	# ---------------------------------------------------------------
	# exit
	#   - call exit method of superclass 
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def exit( self, *objects ):

		# call exit method of superclass 
		PluginWindow.exit(self, *objects)

	# end of exit


	# ---------------------------------------------------------------
	# copyFullPNList
	#   - copies FullPNList to clipboard
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	#def copyFullPNList(self, *objects ):

	#	self.theFullPNListClipBoard = self.theRawFullPNList

	# end of copyFullPNList
	

	# ---------------------------------------------------------------
	# pasteFullPNList
	#   - pastes FullPNList to clipboard
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	#def pasteFullPNList(self, *objects ):

	#	self.theRawFullPNList = self.theFullPNListClipBoard
	#	self.initialize()

	# end of pasteFullPNList


	# ---------------------------------------------------------------
	# addFullPNList
	#   - adds FullPNList to clipboard
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	#def addFullPNList(self, *objects ):

	#	self.theRawFullPNList.extend( self.theFullPNListClipBoard )

	# end of addFullPNList

	# ---------------------------------------------------------------
	# createNewPluginWindow
	#
	# anObject  :  the plugin window that this instance will change to
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	#def createNewPluginWindow( self, anObject ):

	#	aPluginName = anObject.get_name()
	#	self.thePluginManager.createInstance( aPluginName, self.getRawFullPNList() )

	# end of changePluginWindow


	# ---------------------------------------------------------------
	# createLogger
	#   - create Logger of theFullPN
	#
	# *objects : dammy objects
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def createLogger( self, *objects ):
		aLogPolicy = self.theSession.getLogPolicyParameters()
		try:
			for aFullPN in self.theFullPNList():

				aFullPNString = createFullPNString(aFullPN)
			
				# creates loggerstub and call its create method.
				aLoggerStub = self.theSession.createLoggerStub( aFullPNString )
				if aLoggerStub.exists() == FALSE:
					aLoggerStub.create()
					aLoggerStub.setLoggerPolicy( aLogPolicy )

		except:

			# When to create log is failed, display error message on MessageWindow.
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.thePluginManager.printMessage( anErrorMessage )

		# updates fandamental windows.
		self.thePluginManager.updateFundamentalWindows()

	# end of createLogger


	# ---------------------------------------------------------------
	# changeFullPN
	#
	# anObject : the FullID that this instance will show
	#
	# return -> None
	# This method throws exceptions.
	# ---------------------------------------------------------------
	def changeFullPN( self, anObject ):

		index = 0
		for aFullPN in self.theFullPNList():
			aFullPNString = createFullPNString( aFullPN )
			if aFullPNString == anObject.get_name():
				break
			index = index + 1

		self.theSelectedFullPNIndex = index

		self.update()
		self.thePluginManager.updateAllPluginWindow()
		self.thePluginManager.updateFundamentalWindows()

	# end of changeFullPN

	# ========================================================================
	def isStandAlone(self):
		""" returns True if plugin is in a separate window
			False if it is on a BoardWindow
		"""
		return self.getParent().__class__.__name__[:5]!='Board'
		


	# ========================================================================
	def present( self ):
		"""moves this window to the top of desktop.
		if plugin is on BoardWindow, does nothing.
		Returns None
		"""

		if self.isStandAlone():
			self[self.__class__.__name__].present()

	# ========================================================================
	def iconify( self ):
		"""moves this window to the taskbar.
		When it is on Boardwindow, does nothing.
		Returns None
		"""
	
		if self.isStandAlone():
			self[self.__class__.__name__].iconify()

	# ========================================================================
	def move( self, xpos, ypos ):
		"""moves this window on the desktop to (xpos,ypos).
		When it is on Boardwindow, does nothing.
		Returns None
		"""

		if self.isStandAlone():
			self[self.__class__.__name__].move( xpos, ypos)

	# ========================================================================
	def resize( self, width, heigth ):
		"""resizes this window according to width and heigth.
		Returns None
		"""
		self[self.__class__.__name__].resize( width, heigth)


# end of OsogoPluginWindow
