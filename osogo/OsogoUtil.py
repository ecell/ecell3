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
#'Design: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from ecell.ecssupport import *
import string

# ----------------------------------------------------------------------
# Index of stepper's proprety
# ----------------------------------------------------------------------

SETTABLE = 0
GETABLE  = 1

# ----------------------------------------------------------------------
# View types of OsogoPluginWindow
# ----------------------------------------------------------------------

SINGLE   = 0
MULTIPLE = 1

# ----------------------------------------------------------------------
# decodeAttribute
#
# anAttribute : an attribute ( TRUE or FALSE )
#
# return -> '+' or '-'
# ----------------------------------------------------------------------
def decodeAttribute(anAttribute):

	if anAttribute == TRUE:
		return '+'
	else:
		return '-'

# end of decodeAttribute


# ----------------------------------------------------------------------
# convertStringToTuple
#
# aString : a string as below
#           "(a,b,c)"
#
# return -> '+' or '-'
# ----------------------------------------------------------------------
def convertStringToTuple(aString):

	aString = aString[1:-1]
	aList = string.split(aString,',')

	for anIndex in range(0,len(aList)):
		anElement = aList[anIndex]
		anElement = string.strip(anElement)
		try:
			anElement = string.atoi(anElement)
		except:
			try:
				anElement = string.atof(anElement)
			except:
				anElement = anElement[1:-1]

		aList[anIndex] = anElement

	return tuple(aList)

# end of convertStringToTuple


# ----------------------------------------------------------------------
# shortenString
#
# aValue : an original string
# aNumber : the length to cut original string
#
# return -> a shorten string
# ----------------------------------------------------------------------
def shortenString( aValue, aNumber):

	if len( str(aValue) ) > aNumber:
		return aValue[:aNumber] + '...'
	else:
		return aValue
		
# end of shortenString


# ---------------------------------------------------------------
# getFloatFromString
#
# aString  :  a string to convert number
#
# return -> float
# This method throws exception.
# ---------------------------------------------------------------
def getFloatFromString( aString ):

	# If the inputed charactor is not numerical charactors,
	# then throws exception

	# gets the inputerd characters from the GtkEntry. 
	aNumberString = string.strip( aString )

	if len( aNumberString ) == 0:
		# When string is blank, do nothing
		return None
	else:

		# considers the case that character 'e' is included.

		# for example, 'e' or '4e' 
		if string.find(aNumberString, 'e') == len(aNumberString)-1:
			return None

		# for expample, '3e-' or '5e+'
		if len(aNumberString) >= 2 and \
		   ( string.find(aNumberString, 'e-') == len(aNumberString)-2 or \
			string.find(aNumberString, 'e+') == len(aNumberString)-2 ):
			return None

		# for expample, '3e-2'
		if string.find(aNumberString, 'e-') != -1:
			anIndexNumber = aNumberString[string.find(aNumberString,'e-')+2:]
			anIndexNumber = string.atof( anIndexNumber )
			baseNumber =  aNumberString[:string.find(aNumberString,'e-')]
			if len(baseNumber) == 0:
				aNumberString = "1e-%s" %str( int(anIndexNumber) )
			return string.atof( aNumberString )

		# for expample, '5e+6'
		if string.find(aNumberString, 'e+') != -1:
			anIndexNumber = aNumberString[string.find(aNumberString,'e+')+2:]
			anIndexNumber = string.atof( anIndexNumber )
			baseNumber =  aNumberString[:string.find(aNumberString,'e+')]
			if len(baseNumber) == 0:
				aNumberString = "1e+%s" %str( int(anIndexNumber) )
			return string.atof( aNumberString )

		else:
			# When user input some character, tries to convert
			# it to numerical value.
			# following line is throwable except
			return string.atof( aNumberString )

# end of setStepSize


	# ---------------------------------------------------------------
	# updateByTimeOut
	#
	# obj:  textfield
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateByTimeOut( self, obj ):

		#self.update()
		self.update()
		self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )

	# end of updateByTimeOut


	# ---------------------------------------------------------------
	# removeTimeOut
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def removeTimeOut( self ):

		gtk.timeout_remove( self.theTimer )

	# end of removeTimeOut


	# ---------------------------------------------------------------
	# update
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def update( self ):

		#print "MainWindow.update"
		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )
		self.thePluginManager.updateAllPluginWindow()
        
	# end of update

    
	# ---------------------------------------------------------------
	# clickEntityListWindow
	#
	# button_obj : button
	# *objects : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def clickEntityListWindow( self, *objects ):

		#fix me: this part is Root System's bug.
		#if self.theStepperChecker == 1 and self.theEntityChecker == 0:
		if self.theStepperChecker == 1:

			anEntityListWindow = EntityListWindow.EntityListWindow( self )
			self.theEntityListWindowList.append(anEntityListWindow)
			self.theEntityChecker = 1
		
		self.updateFundamentalWindows()


	# end of toggleEntityList


	# ---------------------------------------------------------------
	# toggleLoggerWindowByMenu
	#   - called when logger menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleLoggerWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindowByMenu( self, button_obj ) :

		self.theLoggerWindow.isShown = self['logger_window_menu'].active 
		self.toggleLoggerWindow()

	# end of toggleLoggerWindowByMenu


	# ---------------------------------------------------------------
	# toggleLoggerWindowByButton
	#   - called when logger menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleLoggerWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindowByButton( self, button_obj ) :

		self.theLoggerWindow.isShown = self['logger_togglebutton'].get_active()
		self.toggleLoggerWindow()

	# end of toggleLoggerWindowByButton

	# ---------------------------------------------------------------
	# toggleLoggerWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindow( self ):

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		#if button_obj.get_active() :
		if self.theLoggerWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Logger Window Widget has destroyed,
			# creates new instance of Logger Window Widget.
			# --------------------------------------------------
			if ( self.theLoggerWindow.getExist() == 0 ):
				self.theLoggerWindow.openWindow()
				#self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
				self.theLoggerWindow.update()

			# --------------------------------------------------
			# If instance of Logger Window Widget has not destroyed,
			# calls show method of Logger Window Widget.
			# --------------------------------------------------
			else:
				self.theLoggerWindow['LoggerWindow'].hide()
				self.theLoggerWindow['LoggerWindow'].show_all()

			self['logger_togglebutton'].set_active(TRUE)
			self['logger_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Logger Window Widget has destroyed,
			# do nothing.
			# --------------------------------------------------
			if ( self.theLoggerWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Logger Window Widget has not destroyed,
			# calls hide method of Logger Window Widget.
			# --------------------------------------------------
			else:
				self.theLoggerWindow['LoggerWindow'].hide()

			self['logger_togglebutton'].set_active(FALSE)
			self['logger_window_menu'].set_active(FALSE)

	# end of toggleLoggerWindow



	# ---------------------------------------------------------------
	# toggleStepperWindowByMenu
	#   - called when stepper menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleStepperWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindowByMenu( self, button_obj ) :

		self.theStepperWindow.isShown = self['stepper_window_menu'].active 
		self.toggleStepperWindow()

	# end of toggleStepperWindowByMenu


	# ---------------------------------------------------------------
	# toggleStepperWindowByButton
	#   - called when stepper menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleStepperWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindowByButton( self, button_obj ) :

		self.theStepperWindow.isShown = self['stepper_togglebutton'].get_active()
		self.toggleStepperWindow()

	# end of toggleStepperWindowByButton

	# ---------------------------------------------------------------
	# toggleStepperWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindow( self ):

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		#if button_obj.get_active() :
		if self.theStepperWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Stepper Window Widget has destroyed,
			# creates new instance of Stepper Window Widget.
			# --------------------------------------------------
			if ( self.theStepperWindow.getExist() == 0 ):
				self.theStepperWindow.openWindow()
				#self.theStepperWindow = StepperWindow.StepperWindow( self.theSession , self )
				self.theStepperWindow.update()

			# --------------------------------------------------
			# If instance of Stepper Window Widget has not destroyed,
			# calls show method of Stepper Window Widget.
			# --------------------------------------------------
			else:
				self.theStepperWindow['StepperWindow'].hide()
				self.theStepperWindow['StepperWindow'].show_all()
				self.theStepperWindow.update()

			self['stepper_togglebutton'].set_active(TRUE)
			self['stepper_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Stepper Window Widget has destroyed,
			# do nothing.
			# --------------------------------------------------
			if ( self.theStepperWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Stepper Window Widget has not destroyed,
			# calls hide method of Stepper Window Widget.
			# --------------------------------------------------
			else:
				self.theStepperWindow['StepperWindow'].hide()

			self['stepper_togglebutton'].set_active(FALSE)
			self['stepper_window_menu'].set_active(FALSE)

	# end of toggleStepperWindow


	# ---------------------------------------------------------------
	# toggleMessageWindowByMenu
	#   - called when message menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleMessageWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindowByMenu( self, button_obj ) :

		self.theMessageWindow.isShown = self['message_window_menu'].active 
		self.toggleMessageWindow()

	# end of toggleMessageWindowByMenu


	# ---------------------------------------------------------------
	# toggleMessageWindowByMenu
	#   - called when message button is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleMessageWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindowByButton( self, button_obj ) :

		self.theMessageWindow.isShown = self['message_togglebutton'].get_active()
		self.toggleMessageWindow()

	# end of toggleMessageWindowByButton

	# ---------------------------------------------------------------
	# toggleMessageWindow
	#
	# button_obj : button or menu
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindow( self ) :

		# ------------------------------------------------------
		# button or menu is toggled as active 
		# ------------------------------------------------------
		if self.theMessageWindow.isShown == FALSE:

			# --------------------------------------------------
			# hide handlebox, resize window
			# --------------------------------------------------
			self['handlebox24'].hide()
			self.resize_vertically(0)


			self['message_togglebutton'].set_active(FALSE)
			self['message_window_menu'].set_active(FALSE)

		# ------------------------------------------------------
		# button or menu is toggled as non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# show handlebox, resize window			# 
			# --------------------------------------------------
			self['handlebox24'].show()
			if self.MessageWindow_attached:
			    self.resize_vertically(self.MW_actual_size[1])
			else:
			    self.resize_vertically(0)

			self['message_togglebutton'].set_active(TRUE)
			self['message_window_menu'].set_active(TRUE)


	# end of toggleMessageWindow

	# ---------------------------------------------------------------
	# MW_child_attached
	# called when MessageBox is reatached to MainWindow
	# must resize msgbox scrolledwindow to minimal size
	# and the Mainwindow to extended size
	# ---------------------------------------------------------------
	
	def MW_child_attached(self,obj,obj2):
	    self['scrolledwindow1'].set_size_request(self.MW_minimal_size[0],\
		    self.MW_minimal_size[1])
	    self.resize_vertically(self.MW_actual_size[1])
	    self.MessageWindow_attached=TRUE

	# ---------------------------------------------------------------
	# MW_child_detached
	# called when MessageBox is detached from MainWindow
	# must resize msgbox scrolledwindow to actual size
	# and the Mainwindow to minimalsize
	# ---------------------------------------------------------------
	    
	def MW_child_detached(self,obj,obj2):
	    self['scrolledwindow1'].set_size_request(self.MW_actual_size[0],\
		    self.MW_actual_size[1])
	    self.resize_vertically(0)
	    self.MessageWindow_attached=FALSE
	        
	# ---------------------------------------------------------------
	# openAbout
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openAbout( self, button_obj ):

		anAboutDialog = gnome.ui.About( NAME,
		                                     VERSION,
		                                     COPYRIGHT,
						     DESCRIPTION,
		                                     AUTHORLIST)
		anAboutDialog.set_title( 'about osogo' )
		anAboutDialog.show_all()

	# end of openAbout


	# ---------------------------------------------------------------
	# openPreference
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openPreferences( self, button_obj ):

		#aPropertyBox = gnome.ui.GnomePropertyBox()
		aPropertyBox = gnome.ui.PropertyBox()
		aLabel = gtk.Label( 'NOT IMPLEMENTED YET' )
		aTabLabel = gtk.Label( 'warning' )
		aPropertyBox.append_page( aLabel, aTabLabel )

		#aPropertyBox = gnome.ui.GnomePropertyBox()
		#aLabel = gtk.Label( 'NOT IMPLEMENTED YET' )
		#aTabLabel = gtk.Label( 'warning' )
		#aPropertyBox.append_page( aLabel, aTabLabel )
		#aPropertyBox.hide()
		#aPropertyBox.show_all()
		aMessage = ' Sorry ! Not implemented... [%s]\n' %'06/Mar/2003'
		self.printMessage(aMessage)
		aDialog = ConfirmWindow(0,aMessage,'Sorry!')
		return None

	# end of openPreference


	# ---------------------------------------------------------------
	# initializePaletteWindow
	#
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def initializePaletteWindow( self ):

		self.thePaletteWindow = PaletteWindow.PaletteWindow( self )
		self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

	# end of initializePaletteWindow



	# ---------------------------------------------------------------
	# togglePaletteWindowByMenu
	#   - called when palette menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls togglePalleteWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def togglePaletteWindowByMenu( self, button_obj ) :

		self.thePaletteWindow.isShown = self['palette_window_menu'].active 
		self.togglePaletteWindow()

	# end of togglePaletteWindowByMenu


	# ---------------------------------------------------------------
	# togglePaletteWindowByButton
	#   - called when palette button is toggled.
	#   - sets the "isShown" attribute.
	#   - calls togglePalleteWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def togglePaletteWindowByButton( self, button_obj ) :

		self.thePaletteWindow.isShown = self['palette_togglebutton'].get_active()
		self.togglePaletteWindow()

	# end of togglePaletteWindowByButton


	# ---------------------------------------------------------------
	# togglePaletteWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def togglePaletteWindow( self ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if self.thePaletteWindow.isShown == TRUE:

			if self.thePaletteWindow.getExist() == 0:
				self.initializePaletteWindow()
			else:
				pass	

			self.thePaletteWindow.hide()
			self.thePaletteWindow.show_all()

			self['palette_togglebutton'].set_active(TRUE)
			self['palette_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:
			if self.thePaletteWindow.getExist() == 0:
				pass
			else:
				self.thePaletteWindow.hide()

			self['palette_togglebutton'].set_active(FALSE)
			self['palette_window_menu'].set_active(FALSE)
        
	# end of togglePaletteWindow


	# ---------------------------------------------------------------
	# toggleInterfaceWindowByMenu
	#   - called when interface menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleInterfaceWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindowByMenu( self, *objects ) :

		self.theInterfaceWindow.isShown = self['interface_window_menu'].active 
		self.toggleInterfaceWindow()

	# end of toggleInterfaceWindowByMenu


	# ---------------------------------------------------------------
	# toggleInterfaceWindowByButton
	#   - called when interface button is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleInterfaceWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindowByButton( self, *objects ) :

		self.theInterfaceWindow.isShown = self['interface_togglebutton'].get_active()
		self.toggleInterfaceWindow()

	# end of toggleInterfaceWindowByButton


	# ---------------------------------------------------------------
	# toggleInterfaceWindow
	#   - show or hide InterfaceWindow according to "isShown" attribute.
	#
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindow( self ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if self.theInterfaceWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Interface Window Widget has destroyed,
			# creates new instance of Interface Window Widget.
			# --------------------------------------------------
			if ( self.theInterfaceWindow.getExist() == 0 ):
				self.theInterfaceWindow.openWindow()
				self.theInterfaceWindow.update()

			# --------------------------------------------------
			# If instance of Interface Window Widget has not destroyed,
			# calls show method of Interface Window Widget.
			# --------------------------------------------------
			else:
				self.theInterfaceWindow['InterfaceWindow'].hide()
				self.theInterfaceWindow['InterfaceWindow'].show_all()

			self['interface_togglebutton'].set_active(TRUE)
			self['interface_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# does nothing.
			# --------------------------------------------------
			if ( self.theInterfaceWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Interface Window Widget has not destroyed,
			# calls hide method of Interface Window Widget.
			# --------------------------------------------------
			else:
				self.theInterfaceWindow['InterfaceWindow'].hide()

			self['interface_togglebutton'].set_active(FALSE)
			self['interface_window_menu'].set_active(FALSE)

	# end of toggleInterfaceListWindow


	# ---------------------------------------------------------------
	# saveCellStateToTheFile ( NOT IMPLEMENTED ON 08/Jul/2002)
	#
	# *objects : dammy objects
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def saveCellStateToTheFile( self, *objects ):

		aMessage = ' Sorry ! Not implemented... [%s]\n' %'08/Jul/2002'
		self.printMessage(aMessage)
		aDialog = ConfirmWindow(0,aMessage,'Sorry!')
		return None

	# end of saveCellStateToTheFile


	# ---------------------------------------------------------------
	# printMessage
	#
	# aMessage : message
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def printMessage( self, aMessage ):

		self.theMessageWindow.printMessage( aMessage )

	# end of printMessage


	# ---------------------------------------------------------------
	# updateFundamentalWindows
	#  - update MessageWindow, LoggerWindow, InterfaceWindow
	#  - update status of each menu and button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateFundamentalWindows( self ):

		# -------------------------------------------
		# calls update method of each Window
		# -------------------------------------------

#		self.theMessageWindow.update()
		self.theLoggerWindow.update()
		self.theInterfaceWindow.update()
		self.theStepperWindow.update()

		# -------------------------------------------
		# checks buttons  ane menus
		# -------------------------------------------
		if self.getExist() == FALSE:
			pass
			#self.exit()
		else:

			# checks message button
			if self.theMessageWindow.isShown == TRUE:
				self['message_togglebutton'].set_active(TRUE)
				self['message_window_menu'].set_active(TRUE)
			else:
				self['message_togglebutton'].set_active(FALSE)
				self['message_window_menu'].set_active(FALSE)

			# checks logger button
			if self.theLoggerWindow.isShown == TRUE:
				self['logger_togglebutton'].set_active(TRUE)
				self['logger_window_menu'].set_active(TRUE)
			else:
				self['logger_togglebutton'].set_active(FALSE)
				self['logger_window_menu'].set_active(FALSE)

			# checks stepper button
			if self.theStepperWindow.isShown == TRUE:
				self['stepper_togglebutton'].set_active(TRUE)
				self['stepper_window_menu'].set_active(TRUE)
			else:
				self['stepper_togglebutton'].set_active(FALSE)
				self['stepper_window_menu'].set_active(FALSE)


			# checks interface button
			if self.theInterfaceWindow.isShown == TRUE :
				self['interface_togglebutton'].set_active(TRUE)
				self['interface_window_menu'].set_active(TRUE)
			else:
				self['interface_togglebutton'].set_active(FALSE)
				self['interface_window_menu'].set_active(FALSE)

			if self.theStepperChecker:

				# if palette button pressed
				if self['palette_togglebutton'].get_active():
					if self.thePaletteWindow.isShown == TRUE :
						self['palette_togglebutton'].set_active(TRUE)
						self['palette_window_menu'].set_active(TRUE)
					else:
						self['palette_togglebutton'].set_active(FALSE)
						self['palette_window_menu'].set_active(FALSE)

			# entity window
			# detects the destroyed EntityWindows, and delete them from
			# self.theEntityListWindow.
			aDeleteIndexList = []
			for anIndex in range(0,len(self.theEntityListWindowList)):
				anEntityListWindow = self.theEntityListWindowList[anIndex]

				if anEntityListWindow.getExist() == TRUE:
					anEntityListWindow.update()
				else:
					aDeleteIndexList.append(anIndex)

			aDeleteIndexList.sort()
			aDeleteIndexList.reverse()
			for anIndex in aDeleteIndexList:
				del self.theEntityListWindowList[anIndex]


		# When model file is loaded
		if self.theSession.theModelName != "":
			self.theStepperChecker = TRUE

			self['start_button'].set_sensitive(TRUE)
			self['stop_button'].set_sensitive(TRUE)
			self['step_button'].set_sensitive(TRUE)
			self['entitylist'].set_sensitive(TRUE)
			self['palette_togglebutton'].set_sensitive(TRUE)
			self['palette_window_menu'].set_sensitive(TRUE)
			self['create_new_entity_list_menu'].set_sensitive(TRUE)
			self['load_rule_menu'].set_sensitive(FALSE)
			self['load_script_menu'].set_sensitive(FALSE)
			self['save_model_menu'].set_sensitive(TRUE)

	# end of updateFundamentalWindow


# end of MainWindow




















