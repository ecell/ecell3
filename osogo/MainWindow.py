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
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


#from Window import *
from OsogoWindow import *

from main import *
#from Plugin import *
from OsogoPluginManager import *

import gnome.ui
import gtk
import GTK

import MessageWindow
import PaletteWindow 
import EntryListWindow
import LoggerWindow
import InterfaceWindow 

import string

#
#import pyecell module
#
import ecell.ecs
import ecell.Session
from ecell.ecssupport import *
from ecell.ECS import *


NAME        = 'gecell (Osogo)'
VERSION     = '0.0'
COPYRIGHT   = '(C) 2001-2002 Keio University'
AUTHORLIST  = [
    'Design: Kenta Hashimoto <kem@e-cell.org>',
    'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
    'Programming: Yuki Fujita',
    'Yoshiya Matsubara',
    'Yuusuke Saito',
    'Masahiro Sugimoto <sugi@e-cell.org>'
    ]
DESCRIPTION = 'Osogo is a simulation session monitoring module for E-CELL SE Version 3'


# ---------------------------------------------------------------
# MainWindow -> OsogoWindow
#   - manages MainWindow
# ---------------------------------------------------------------
class MainWindow(OsogoWindow):

	RuleFileExtension = 'eml'
	ScriptFileExtension = 'py'
	CellStateFileExtension = 'cs'

	# ---------------------------------------------------------------
	# Constructor
	#   - creates MessageWindow 
	#   - creates LoggerWindow 
	#   - creates InterfaceWindow 
	#   - creates PalleteWindow 
	#   - creates PluginManager 
	#   - creates Session 
	#   - adds signal to handers
	#   - creates ScriptFileSelection
	#   - creates RuleFileSelection
	#   - creates CellStateFileSelection
	#   - initialize Session
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self ):
	
		OsogoWindow.__init__( self, self, 'MainWindow.glade' )
		self.openWindow()

	# end of init

	def openWindow( self ):
		OsogoWindow.openWindow(self)

		# -------------------------------------
		# creates MessageWindow 
		# -------------------------------------
		self.theMessageWindow = MessageWindow.MessageWindow( self )
		self.theMessageWindow.openWindow()

		# -------------------------------------
		# creates Session
		# -------------------------------------
		self.theSession = ecell.Session.Session( ecell.ecs.Simulator() )
		self.theSession.setPrintMethod( self.theMessageWindow.printMessage )

		# -------------------------------------
		# creates LoggerWindow
		# -------------------------------------
		self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
		self.theLoggerWindow.openWindow()

		# -------------------------------------
		# creates InterfaceWindow
		# -------------------------------------
		self.theInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
		self.theInterfaceWindow.openWindow()

		# -------------------------------------
		# creates PluginManager
		# -------------------------------------
		#self.thePluginManager = OsogoPluginManager( self.theSession, 
		#        self.theLoggerWindow, self.theInterfaceWindow, self.theMessageWindow )

		self.thePluginManager = OsogoPluginManager( self )
		self.thePluginManager.loadAll()

		# -------------------------------------
		# creates PaletteManager
		# -------------------------------------
		self.thePaletteWindow = PaletteWindow.PaletteWindow( self )
		self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

		self.theEntryChecker = 0
		self.theStepperChecker = 0

		self.theUpdateInterval = 150
		self.theStepSize = 1
		self.theStepType = 0
		self.theRunningFlag = 0


		self.theEntryListWindowList = []

		self.theHandlerMap = \
		  { 'load_rule_menu_activate'              : self.openRuleFileSelection ,
			'load_script_menu_activate'            : self.openScriptFileSelection ,
			'save_cell_state_menu_activate'        : self.saveCellStateToTheFile ,
			'save_cell_state_as_menu_activate'     : self.openSaveCellStateFileSelection ,
			'exit_menu_activate'                   : self.exit ,
			'message_window_menu_activate'         : self.toggleMessageWindowByMenu ,
			'interface_window_menu_activate'       : self.toggleInterfaceWindowByMenu ,
			'palette_window_menu_activate'         : self.togglePaletteWindowByMenu ,
			'create_new_entry_list_menu_activate'  : self.clickEntryListWindow ,
			'logger_window_menu_activate' : self.toggleLoggerWindowByMenu ,
			'preferences_menu_activate'            : self.openPreferences ,
			'about_menu_activate'                  : self.openAbout ,
			'start_button_clicked'                 : self.startSimulation ,
			'stop_button_clicked'                  : self.stopSimulation ,
			'step_button_clicked'                  : self.stepSimulation ,
			'input_step_size'                      : self.setStepSize ,
			'step_sec_toggled'                     : self.changeStepType ,
			'entrylist_clicked'       : self.clickEntryListWindow ,
			'logger_togglebutton_toggled'          : self.toggleLoggerWindowByButton ,
			'palette_togglebutton_toggled'         : self.togglePaletteWindowByButton ,
			'message_togglebutton_toggled'         : self.toggleMessageWindowByButton ,
			'interface_togglebutton_toggled'       : self.toggleInterfaceWindowByButton ,
			'logo_button_clicked'                  : self.openAbout,
		}
		self.addHandlers( self.theHandlerMap )


		# -------------------------------------
		# create Save Cell State File Selection 
		# -------------------------------------
		self.theSaveFileSelection = gtk.GtkFileSelection( 'Select Rule File for Saving' )
		self.theSaveFileSelection.ok_button.connect('clicked', self.saveCellState)
		self.theSaveFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theSaveFileSelection.complete( '*.' + self.CellStateFileExtension )

		# -------------------------------------
		# initialize for run method 
		# -------------------------------------
		self.theSession.theSimulator.setPendingEventChecker( gtk.events_pending )
		self.theSession.theSimulator.setEventHandler( gtk.mainiteration  )
		self['ecell_logo_toolbar'].set_style( GTK.TOOLBAR_ICONS )
		self.setInitialWidgetStatus()
		self.updateBasicWindows()

	# end of __init__


	def setUnSensitiveMenu( self ):
		self['palette_window_menu'].set_sensitive(0)
		self['create_new_entry_list_menu'].set_sensitive(0)

	# end of setUnSensitiveMenu


	# ---------------------------------------------------------------
	# setInitialWidgetStatus
	#   - set initial status to all of the widgets on this window
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setInitialWidgetStatus( self ):
		self['start_button'].set_sensitive(0)
		self['stop_button'].set_sensitive(0)
		self['step_button'].set_sensitive(0)
		self['entrylist'].set_sensitive(0)
		self['palette_togglebutton'].set_sensitive(0)
		self['palette_window_menu'].set_sensitive(0)
		self['create_new_entry_list_menu'].set_sensitive(0)

		self.setUnSensitiveMenu()

	# end of setInitialWidgetStatus


	# ---------------------------------------------------------------
	# closeParentWindow
	#
	# anObject: a reference to widget 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def closeParentWindow( self, anObject ):

		aParentWindow = anObject.get_parent_window()
		aParentWindow.hide()

	# end of closeParentWindow


	# ---------------------------------------------------------------
	# openRuleFileSelection
	#
	# anObject: a reference to widget
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openRuleFileSelection( self, anObject ) :

		self.theRuleFileSelection = gtk.GtkFileSelection( 'Select Rule File' )
		self.theRuleFileSelection.set_modal(gtk.TRUE)
		self.theRuleFileSelection.ok_button.connect('clicked', self.loadRule)
		self.theRuleFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theRuleFileSelection.complete( '*.' + self.RuleFileExtension )
		self.theRuleFileSelection.show_all()

	# end of openRuleFileSelection


	# ---------------------------------------------------------------
	# loadRule
	#
	# button_obj: reference to button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def loadRule( self, button_obj ) :

		try:
			aFileName = self.theRuleFileSelection.get_filename()

			if os.path.isfile( aFileName ):
				pass
			else:
				aMessage = ' Error ! No such file. \n[%s]' %aFileName
				self.printMessage(aMessage)
				aDialog = ConfirmWindow(0,aMessage,'Error!')
				return None

			self.theRuleFileSelection.hide()
			self.theSession.printMessage( 'loading rule file %s\n' % aFileName)
			aModelFile = open( aFileName )
			self.theSession.loadModel( aModelFile )
			aModelFile.close()
			self.theSession.theSimulator.initialize()
			self.update()
			self.updateBasicWindows()

		except:
			self.printMessage(' can\'t load [%s]' %aFileName)
			aErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage("-----------")
			self.printMessage(aErrorMessage)
			self.printMessage("-----------")
		else:
			self.theStepperChecker = 1
			self['start_button'].set_sensitive(1)
			self['stop_button'].set_sensitive(1)
			self['step_button'].set_sensitive(1)
			self['entrylist'].set_sensitive(1)
			self['palette_togglebutton'].set_sensitive(1)
			self['palette_window_menu'].set_sensitive(1)
			self['create_new_entry_list_menu'].set_sensitive(1)
			self['load_rule_menu'].set_sensitive(0)
			self['load_script_menu'].set_sensitive(0)
			#self.setUnSensitiveMenu()

	# end of loadRule


	# ---------------------------------------------------------------
	# openScriptFileSelection
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openScriptFileSelection( self, anObject ) :

		self.theScriptFileSelection = gtk.GtkFileSelection( 'Select Script File' )
		self.theScriptFileSelection.set_modal(gtk.TRUE)
		self.theScriptFileSelection.ok_button.connect('clicked', self.loadScript)
		self.theScriptFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theScriptFileSelection.complete( '*.' + self.ScriptFileExtension )
		self.theScriptFileSelection.show_all()
        
	# end of openScriptFileSelection


	# ---------------------------------------------------------------
	# loadScript
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def loadScript( self, anObject ):

		try:
			aFileName = self.theScriptFileSelection.get_filename()
			self.theScriptFileSelection.hide()
			self.theSession.printMessage( 'loading script file %s\n' % aFileName )
			aGlobalNameMap = { 'theMainWindow' : self }
			execfile( aFileName, aGlobalNameMap )
			self.update()
			self.updateBasicWindows()
		except:
			import sys
			import traceback
			self.printMessage(' can\'t load [%s]' %aFileName)
			aErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage("-----------")
			self.printMessage(aErrorMessage)
			self.printMessage("-----------")
		else:
			self.theStepperChecker = 1
			self['start_button'].set_sensitive(1)
			self['stop_button'].set_sensitive(1)
			self['step_button'].set_sensitive(1)
			self['entrylist'].set_sensitive(1)
			self['palette_togglebutton'].set_sensitive(1)
			self['create_new_entry_list_menu'].set_sensitive(1)
			self['load_rule_menu'].set_sensitive(0)
			self['load_script_menu'].set_sensitive(0)
			self.setUnSensitiveMenu()

	# end of loadScript


	# ---------------------------------------------------------------
	# openSaveCellStateFileSelection
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openSaveCellStateFileSelection( self, anObject ) :

		#self.theSaveFileSelection.show_all()
		aMessage = ' Sorry ! Not implemented... [%s]\n' %'08/Jul/2002'
		self.printMessage(aMessage)
		aDialog = ConfirmWindow(0,aMessage,'Sorry!')
		return None

	# end of openSaveCellStateFileSelection


	# ---------------------------------------------------------------
	# saveCellState ( NOT IMPLEMENTED ON 27/Jul/2002)
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def saveCellState ( self, anObject ) :

		pass

	# end of saveCellState

	# ---------------------------------------------------------------
	# exit
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def exit( self, anObject=None ):

		aMessage = '\n Are you sure you want to quit? \n'
		aDialog = ConfirmWindow(1,aMessage,'?')

		try:
			if aDialog.return_result() == 0:
				mainQuit()
			else:
				if self.getExist():
					pass
				else:
					OsogoWindow.openWindow(self)
		except:
			pass
        
	# end of exit


	# ---------------------------------------------------------------
	# startSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def startSimulation( self, obj ) :

		try:

			self.theRunningFlag = 1
			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			self.theSession.printMessage( "Start\n" )
			self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.updateByTimeOut, 0)
			self.theLoggerWindow.update()
			self.theSession.run()
			self.removeTimeOut()

		except:
			self.theRunningFlag = 0
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)

		else:
			pass

	# end of startSimulation
            
	# ---------------------------------------------------------------
	# stopSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def stopSimulation( self, obj ) :

		try:
			if self.theRunningFlag:
				self.theSession.stop()
				self.theSession.printMessage( "Stop\n" )
				self.removeTimeOut()
				self.update()
				self.updateBasicWindows()
				self.theLoggerWindow.update()

		except:
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)

		else:
			self.theRunningFlag = 0

		self.updateBasicWindows()

	# end of stopSimulation


	# ---------------------------------------------------------------
	# stepSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def stepSimulation( self, obj ) : 

		try:


			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			self.theSession.printMessage( "Step\n" )
			self['step_combo_entry'].set_text( str( self.theStepSize ) )
			self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )
			if self.theStepType == 0:
				self.theSession.run( self.theStepSize )
			else:
				self.theSession.step( self.theStepSize )
			self.removeTimeOut()
			self.update()
			self.updateBasicWindows()
			self.theLoggerWindow.update()


		except:
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)

		else:
			pass

	# end of stepSimulation

            
	# ---------------------------------------------------------------
	# changeStepType
	#
	# anObject:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def changeStepType ( self, anObject ):

		self.theStepType = 1 - self.theStepType

	# end of changeStepType


	# ---------------------------------------------------------------
	# setStepSize
	#
	# anObject:  textfield
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setStepSize( self, anObject ):

		try:

			aNumberString = anObject.get_text()

			if len( aNumberString ):
				self.theStepSize = string.atof( aNumberString )
			else:
				self.theStepSize = 1

		except:
			aMessage = "[%s] can't be changed float number." %anObject.get_text()
			self.printMessage(aMessage)
			aDialog = ConfirmWindow(0,aMessage,'Error!')

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


		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )
		self.thePluginManager.updateAllPluginWindow()
        
	# end of update

    
	# ---------------------------------------------------------------
	# clickEntryListWindow
	#
	# button_obj : button
	# *objects : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def clickEntryListWindow( self, *objects ):

		#fix me: this part is Root System's bug.
		#if self.theStepperChecker == 1 and self.theEntryChecker == 0:
		if self.theStepperChecker == 1:

			anEntryListWindow = EntryListWindow.EntryListWindow( self )
			self.theEntryListWindowList.append(anEntryListWindow)
			self.theEntryChecker = 1
		
		self.updateBasicWindows()


	# end of toggleEntryList


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

		self.theLoggerWindow.isShown = self['loggertogglebutton'].get_active()
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
		if self.theLoggerWindow.isShown == gtk.TRUE:

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

			self['loggertogglebutton'].set_active(gtk.TRUE)
			self['logger_window_menu'].set_active(gtk.TRUE)

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

			self['loggertogglebutton'].set_active(gtk.FALSE)
			self['logger_window_menu'].set_active(gtk.FALSE)

	# end of toggleLoggerWindow


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
		#if button_obj.get_active() :
		#if MessageWindowWillBeShown == gtk.TRUE:
		if self.theMessageWindow.isShown == gtk.TRUE:

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# creates new instance of Message Window Widget.
			# --------------------------------------------------
			if ( self.theMessageWindow.getExist() == 0 ):
				self.theMessageWindow.openWindow()

			# --------------------------------------------------
			# If instance of Message Window Widget has not destroyed,
			# calls show method of Message Window Widget.
			# --------------------------------------------------
			else:
				self.theMessageWindow['MessageWindow'].hide()
				self.theMessageWindow['MessageWindow'].show_all()

			self['message_togglebutton'].set_active(gtk.TRUE)
			self['message_window_menu'].set_active(gtk.TRUE)

		# ------------------------------------------------------
		# button or menu is toggled as non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# does nothing.
			# --------------------------------------------------
			if ( self.theMessageWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Message Window Widget has not destroyed,
			# calls hide method of Message Window Widget.
			# --------------------------------------------------
			else:
				self.theMessageWindow['MessageWindow'].hide()

			self['message_togglebutton'].set_active(gtk.FALSE)
			self['message_window_menu'].set_active(gtk.FALSE)


	# end of toggleMessageWindow


	# ---------------------------------------------------------------
	# openAbout
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openAbout( self, button_obj ):

		anAboutDialog = gnome.ui.GnomeAbout( NAME,
		                                     VERSION,
		                                     COPYRIGHT,
		                                     AUTHORLIST,
		                                     DESCRIPTION )
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

		aPropertyBox = gnome.ui.GnomePropertyBox()
		aLabel = gtk.GtkLabel( 'NOT IMPLEMENTED YET' )
		aTabLabel = gtk.GtkLabel( 'warning' )
		aPropertyBox.append_page( aLabel, aTabLabel )

		aPropertyBox.hide()
		aPropertyBox.show_all()

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
		if self.thePaletteWindow.isShown == gtk.TRUE:

			if self.thePaletteWindow.getExist() == 0:
				self.initializePaletteWindow()
			else:
				pass	

			self.thePaletteWindow.hide()
			self.thePaletteWindow.show_all()

			self['palette_togglebutton'].set_active(gtk.TRUE)
			self['palette_window_menu'].set_active(gtk.TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:
			if self.thePaletteWindow.getExist() == 0:
				pass
			else:
				self.thePaletteWindow.hide()

			self['palette_togglebutton'].set_active(gtk.FALSE)
			self['palette_window_menu'].set_active(gtk.FALSE)
        
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
		if self.theInterfaceWindow.isShown == gtk.TRUE:

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

			self['interface_togglebutton'].set_active(gtk.TRUE)
			self['interface_window_menu'].set_active(gtk.TRUE)

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

			self['interface_togglebutton'].set_active(gtk.FALSE)
			self['interface_window_menu'].set_active(gtk.FALSE)

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
	# updateBasicWindows
	#  - update MessageWindow, LoggerWindow, InterfaceWindow
	#  - update status of each menu and button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateBasicWindows( self ):


		# -------------------------------------------
		# calls update method of each Window
		# -------------------------------------------

		self.theMessageWindow.update()
		self.theLoggerWindow.update()
		self.theInterfaceWindow.update()


		for anEntryListWindow in self.theEntryListWindowList:
			anEntryListWindow.update()

		# -------------------------------------------
		# checks buttons  ane menus
		# -------------------------------------------
		if self.getExist() == gtk.FALSE:
			pass
			#self.exit()
		else:

			# checks message button
			if self.theMessageWindow.isShown == gtk.TRUE:
				self['message_togglebutton'].set_active(gtk.TRUE)
				self['message_window_menu'].set_active(gtk.TRUE)
			else:
				self['message_togglebutton'].set_active(gtk.FALSE)
				self['message_window_menu'].set_active(gtk.FALSE)

			# checks logger button
			if self.theLoggerWindow.isShown == gtk.TRUE:
				self['loggertogglebutton'].set_active(gtk.TRUE)
				self['logger_window_menu'].set_active(gtk.TRUE)
			else:
				self['loggertogglebutton'].set_active(gtk.FALSE)
				self['logger_window_menu'].set_active(gtk.FALSE)


			# checks interface button
			if self.theInterfaceWindow.isShown == gtk.TRUE :
				self['interface_togglebutton'].set_active(gtk.TRUE)
				self['interface_window_menu'].set_active(gtk.TRUE)
			else:
				self['interface_togglebutton'].set_active(gtk.FALSE)
				self['interface_window_menu'].set_active(gtk.FALSE)

			if self.theStepperChecker:

				# if palette button pressed
				if self['palette_togglebutton'].get_active():
					if self.thePaletteWindow.isShown == gtk.TRUE :
						self['palette_togglebutton'].set_active(gtk.TRUE)
						self['palette_window_menu'].set_active(gtk.TRUE)
					else:
						self['palette_togglebutton'].set_active(gtk.FALSE)
						self['palette_window_menu'].set_active(gtk.FALSE)

	# end of updateBasicWindow



# end of MainWindow

