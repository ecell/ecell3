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
#             'Yuusuke Saito'
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

from PluginWindow import *

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
	def __init__( self, dirname, data, pluginmanager, root=None ):

		self.theFullPNListClipBoard = []

		#PluginWindow.__init__( self, dirname, data, pluginmanager, root )
		PluginWindow.__init__( self, dirname, pluginmanager, root )

		self.theSession = self.thePluginManager.theSession 
		self.theRawFullPNList = data
		self.theTitle = pluginmanager.theInterfaceWindow.theTitle
        	
	# end of __init__


	# ---------------------------------------------------------------
	# initialize ( overrides PluginWindow )
	#   - creates popup menu
	#
	# aRoot          : root property
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def initialize( self, aRoot=None ):

		aMenuWindow = Window( 'PluginWindowPopupMenu.glade', aRoot='menu' )
		self.thePopupMenu = aMenuWindow['menu']

		if aRoot != None:
			self.theClassName = aRoot
			self[self.theClassName].connect( 'button_press_event', self.popupMenu )
			aMenuWindow.addHandlers( { 'copy_fullpnlist'  : self.copyFullPNList,
				'paste_fullpnlist' : self.pasteFullPNList,
				'add_fullpnlist'   : self.addFullPNList
				} )
		#        else root !='top_vbox':
		else:
			self.getWidget( self.theClassName )['title'] = self.theTitle

	# end of initialize


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

		return self.supplementFullPN( self.theRawFullPNList[0] )

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
	# theAttributeMap
	#   - return an attribute map
	#
	# return -> attribute map 
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def theAttributeMap( self ):

		aMap = {}
		for aFullPN in self.theRawFullPNList:
			aFullID = convertFullPNToFullID( aFullPN )
			aPropertyName = aFullPN[PROPERTY]
			aPropertyListFullPN = convertFullIDToFullPN( aFullID, 'PropertyList' )
			aPropertyList = self.theSession.theSimulator.getProperty( createFullPNString( aPropertyListFullPN ) )
			aAttributeListFullPN = convertFullIDToFullPN( aFullID, 'PropertyAttributes')
			aAttributeList = self.theSession.theSimulator.getProperty( createFullPNString( aAttributeListFullPN ) )
			num = 0
			for aProperty in aPropertyList:
				aPropertyFullPN = convertFullIDToFullPN( aFullID, aProperty )
				aMap[ aPropertyFullPN ] = aAttributeList[ num ]
				num += 1
		return aMap

	# end of theAttributeMap

        
	# ---------------------------------------------------------------
	# getAttribute
	#   - return an attribute 
	#     If there is no attribute on my attribute map, then return 99
	#
	# aFullPN : FullPN
	# return -> attribute map 
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def getAttribute( self, aFullPN ):

		aMap = self.theAttributeMap()
		if aMap.has_key( aFullPN ):
			return aMap[ aFullPN ]
		else:
			return 99

	# end of getAttribute


	# ---------------------------------------------------------------
	# getValue from the session.simulator
	#   - return a value
	#
	# aFullPN : FullPN
	# return -> attribute map 
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def getValue( self, aFullpn ):

		aValueList = self.theSession.theSimulator.getProperty( createFullPNString( aFullpn ) )
		return aValueList[0]


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
	def setValue( self, fullpn, aValue ):

		if self.getAttribute( fullpn ) == 3:
			#aValue = value

			aValueList = aValue

			if type(value) != type((0,)):
				aValueList  = (aValueList,)

			aValueList = (value,)

			# the 2nd argument of theSimulator.setPropety must be tuple
			self.theSession.theSimulator.setProperty( createFullPNString( fullpn ), aValueList )
			self.thePluginManager.updateAllPluginWindow()
		else:
			aFullPNString = createFullPNString( fullpn )
			self.theSession.printMessage('%s is not settable\n' % aFullPNString )

	# end of setValue


	# ---------------------------------------------------------------
	# exit
	#   - call exit method of superclass 
	#   - clear the selected list of InterfaceWindow
	#
	# *objects  : dammy element of arguments
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def exit( self, *objects ):

		PluginWindow.exit(self, objects)
		self.thePluginManager.theInterfaceWindow.theSelectedRow = -1

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
		#ViewWindow.theFullPNListClipBoard = self.theRawFullPNList
		self.theFullPNListClipBoard = self.theRawFullPNList
		#print 'copy :',
		#print ViewWindow.theFullPNListClipBoard

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
		#self.theRawFullPNList = ViewWindow.theFullPNListClipBoard
		self.theRawFullPNList = self.theFullPNListClipBoard
		self.initialize()
		#print 'paste :',
		#print self.theRawFullPNList

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
		#self.theRawFullPNList.extend( ViewWindow.theFullPNListClipBoard )
		self.theRawFullPNList.extend( self.theFullPNListClipBoard )
		#print 'add : ',
		#print self.theRawFullPNList

	# end of addFullPNList


# end of OsogoPluginWindow
