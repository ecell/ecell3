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
from ecell.ECS import *
import string
import sys
from ecell.ecssupport import *

#from PluginWindow import *
from ecell.PluginWindow import *

class OsogoPluginWindow(PluginWindow):

	'''
	self.theRawFullPNList : [ FullPN1, FullID2, FullPN3, , , ]
	theFullPNList()       : [ FullPN1, FullPN2, FullPN3, , , ]
	theFullIDList()       : [ FullID1, FullID2, FullID3, , , ]
	theFullPN()           : FullPN1
	theFullID()           : FullID1
	'''

	# ---------------------------------------------------------------
	# Constructor
	#   - initializes theFullListClipBoard
	#   - sets the session reference
	#   - sets the RowFullPNList
	#   - sets the title from InterfaceWindow
	#
	# aDirname       : directory name includes plugin module 
	# aData          : RawFullPNList
	# aPluginManager : PluginManager
	# aRoot          : root property
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, dirname, aData, pluginmanager, aRoot=None ):

		self.theFullPNListClipBoard = []
		self.theSelectedFullPNIndex = 0

		PluginWindow.__init__( self, dirname, pluginmanager, aRoot )
		PluginWindow.openWindow( self )

		self.theSession = self.thePluginManager.theMainWindow.theSession 
		self.theRawFullPNList = aData

		self.theTitle = self.__class__.__name__

		if aRoot == None:
			#self.thePopupMenu = OsogoMenu( pluginmanager, self )
			#self.getWidget( self.theClassName ).connect( 'button_press_event', self.popupMenu )
			self.getWidget( self.theClassName )['title'] = self.theTitle

	# end of __init__


	# ---------------------------------------------------------------
	# addPopupMenu
	#   - creates instance of popup menu 
	#   - registers singlal to show the popup menu
	#
	# aFullPNMode    : 1  appends FullPN  (default)
	#                  0  doesn't append FullPN
	# aPluginMode    : 1  appends PluginMenu (default)
	#                  0  doesn't append PluginMenu
	# aLoggerMode    : 1  appends 'add logger list' (default)
	#                  0  doesn't append 'add logger list'
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def addPopupMenu( self, aFullPNMode=1, aPluginMode=1, aLoggerMode=1 ):

		# creates instance of popup menu 
		self.thePopupMenu = OsogoPluginWindowPopupMenu( self.thePluginManager, self,
		                                                aFullPNMode, aPluginMode, aLoggerMode )

		# registers singlal to show the popup menu
		self.getWidget( self.theClassName ).connect( 'button_press_event', self.popupMenu )

	# end of addPopupMenu


	def getRawFullPNList( self ):
		return self.theRawFullPNList 


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

		if anEvent.button == 3:
			self.thePopupMenu.popup( None, None, None, 1, 0 )

	# end of poppuMenu


	# ---------------------------------------------------------------
	# theFullPNList
	#   - return FullPNList
	#
	# return -> FullPNList
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def theFullPNList( self ):

		return map( self.supplementFullPN, self.theRawFullPNList )

	# end of theFullPNList


	# ---------------------------------------------------------------
	# theFullIDList
	#   - return FullIDList
	#
	# return -> FullIDList
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def theFullIDList( self ):

		return map( convertFullPNToFullID, self.theRawFullPNList )

	# end of theFullIDList


	# ---------------------------------------------------------------
	# theFullPN
	#   - return FullPN
	#
	# return -> FullPN
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def theFullPN( self ):

		return self.supplementFullPN( self.theRawFullPNList[self.theSelectedFullPNIndex] )

	# end of theFullPN


	# ---------------------------------------------------------------
	# theFullID
	#   - return FullID
	#
	# return -> FullID
	# This method is throwable exception.
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
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def supplementFullPN( self, aFullPN ):

		if aFullPN[PROPERTY] != '' :
			return aFullPN
		else :
			if aFullPN[TYPE] == SUBSTANCE :
				aPropertyName = 'Quantity'
			elif aFullPN[TYPE] == REACTOR :
				aPropertyName = 'Activity'
			elif aFullPN[TYPE] == SYSTEM :
				aPropertyName = 'Activity'
			aNewFullPN = convertFullIDToFullPN( convertFullPNToFullID(aFullPN), aPropertyName )
			return aNewFullPN

	# end of supplementFullPN


	# ---------------------------------------------------------------
	# getValue from the session.simulator
	#   - return a value
	#
	# aFullPN : FullPN
	# return -> attribute map 
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def getValue( self, aFullPN ):

		return self.theSession.theSimulator.getEntityProperty( createFullPNString( aFullPN ) )
		#aValueList = self.theSession.theSimulator.getEntityProperty( createFullPNString( aFullPN ) )
		#return aValueList[0]

	# getValue


	# ---------------------------------------------------------------
	# setValue 
	#   - sets value to the session.simulator
	#
	# aFullPN : FullPN
	# aValue  : one element or tuple
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setValue( self, aFullPN, aValue ):

		aFullID = convertFullPNToFullID( aFullPN )
		aFullPNwithProperty = convertFullIDToFullPN( aFullID, 'PropertyList' )
		aFullPNwithPropertyString = createFullPNString( aFullPNwithProperty )
		aPropertyList = self.theSession.theSimulator.getEntityProperty( aFullPNwithPropertyString )

		for aProperty in aPropertyList:
			# if proprety matches and settable flag is true
			if aProperty[0] == aFullPN[-1] :
				if aProperty[1] == TRUE:
					self.theSession.theSimulator.setEntityProperty( createFullPNString( aFullPN ), aValue )
					self.thePluginManager.updateAllPluginWindow()
					self.thePluginManager.updateFundamentalWindows()
					return None
				else:
					aFullPNString = createFullPNString( aFullPN )
					self.theSession.printMessage('%s is not settable\n' % aFullPNString )
					return None

		aFullPNString = createFullPNString( aFullPN )
		self.theSession.printMessage('proprety of %s is wrong\n' %aFullPNString )


	# end of setValue


	# ---------------------------------------------------------------
	# exit
	#   - call exit method of superclass 
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method is throwable exception.
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
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def copyFullPNList(self, *objects ):

		self.theFullPNListClipBoard = self.theRawFullPNList

	# end of copyFullPNList
	

	# ---------------------------------------------------------------
	# pasteFullPNList
	#   - pastes FullPNList to clipboard
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def pasteFullPNList(self, *objects ):

		self.theRawFullPNList = self.theFullPNListClipBoard
		self.initialize()

	# end of pasteFullPNList


	# ---------------------------------------------------------------
	# addFullPNList
	#   - adds FullPNList to clipboard
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def addFullPNList(self, *objects ):

		self.theRawFullPNList.extend( self.theFullPNListClipBoard )

	# end of addFullPNList

	# ---------------------------------------------------------------
	# createLogger
	#   - changes this instance to other plugin window
	#
	# anObject  :  the plugin window that this instance will change to
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def changePluginWindow( self, anObject ):

		aPluginName = anObject.get_name()
		self.thePluginManager.createInstance( aPluginName, self.getRawFullPNList() )
		self.exit()

	# end of changePluginWindow


	# ---------------------------------------------------------------
	# createLogger
	#   - create Logger of theFullPN
	#
	# *objects : dammy objects
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def createLogger( self, *objects ):

		self.theSession.createLogger( createFullPNString(self.theFullPN()) )
		self.thePluginManager.updateFundamentalWindows()

	# end of createLogger


	# ---------------------------------------------------------------
	# changeFullPN
	#
	# anObject : the FullID that this instance will show
	#
	# return -> None
	# This method is throwable exception.
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


# end of OsogoPluginWindow


# ----------------------------------------------------------
# OsogoPluginPopupMenu -> GtkMenu
#   - popup menu used by plugin menu
# ----------------------------------------------------------
class OsogoPluginWindowPopupMenu( GtkMenu ):

	# ----------------------------------------------------------
	# Constructor
	#   - added PluginManager reference
	#   - added OsogoPluginWindow reference
	#   - acreates all menus
	#
	# aPluginManager : reference to PluginManager
	# aParent        : aOsogoPluginWindow
	# aFullPNMode    : 1  appends FullPN  (default)
	#                  0  doesn't append FullPN
	# aPluginMode    : 1  appends PluginMenu (default)
	#                  0  doesn't append PluginMenu
	# aLoggerMode    : 1  appends 'add logger list' (default)
	#                  0  doesn't append 'add logger list'
	#
	# return -> None
	# This method is throwabe exception.
	# ----------------------------------------------------------
	def __init__( self, aPluginManager, aParent, aFullPNMode=1, aPluginMode=1, aLoggerMode=1 ):

		GtkMenu.__init__(self)

		self.theLoggerString = "add logger list"

		self.theParent = aParent
		self.thePluginManager = aPluginManager
		self.theMenuItem = {}

		# ------------------------------------------
		# adds FullPNString
		# ------------------------------------------
		# if appen FullPN Mode
		if aFullPNMode == 1: 
			for aFullPN in self.theParent.theFullPNList():

				aFullPNString = createFullPNString( aFullPN )
				self.theMenuItem[aFullPNString]= GtkMenuItem(aFullPNString)
				self.theMenuItem[aFullPNString].connect('activate', self.theParent.changeFullPN )
				self.theMenuItem[aFullPNString].set_name(aFullPNString)
				self.append( self.theMenuItem[aFullPNString] )

			self.append( gtk.GtkMenuItem() )

		# ------------------------------------------
		# adds plugin window
		# ------------------------------------------
		# if appen Plugin Mode
		if aPluginMode == 1:
			for aPluginMap in self.thePluginManager.thePluginMap.keys():
				self.theMenuItem[aPluginMap]= GtkMenuItem(aPluginMap)
				self.theMenuItem[aPluginMap].connect('activate', self.theParent.changePluginWindow )
				self.theMenuItem[aPluginMap].set_name(aPluginMap)

				self.append( self.theMenuItem[aPluginMap] )

			self.append( gtk.GtkMenuItem() )

		# ------------------------------------------
		# adds creates logger
		# ------------------------------------------
		# if appen Logger Mode
		if aLoggerMode == 1:
			self.theMenuItem[self.theLoggerString]= GtkMenuItem(self.theLoggerString)
			self.theMenuItem[self.theLoggerString].connect('activate', self.theParent.createLogger )
			self.theMenuItem[self.theLoggerString].set_name(self.theLoggerString)
			self.append( self.theMenuItem[self.theLoggerString] )


	# end of __init__


	# ---------------------------------------------------------------
	# popup
	#    - sets unsensitive following menus.
	#         1) the parent's plugin menu 
	#         2) the current FULLPN menu
	#    - shows this popup memu
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def popup(self, pms, pmi, func, button, time):

		# sets unsensitive following menus.
		for key in self.theMenuItem:
			if ( key == self.theParent.__class__.__name__ ) or ( key == createFullPNString(self.theParent.theFullPN()) ):
				self.theMenuItem[key].set_sensitive(0)
			else:
				self.theMenuItem[key].set_sensitive(1)
		
		# shows this popup memu
		GtkMenu.popup(self, pms, pmi, func, button, time)
		self.show_all(self)

	# end of poup


# end of OsogoPluginWindowPopupMenu



