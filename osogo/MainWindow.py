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
    'Yuusuke Saito'
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
	
		OsogoWindow.__init__( self, 'MainWindow.glade' )
		OsogoWindow.openWindow(self)

		# -------------------------------------
		# creates MessageWindow 
		# -------------------------------------
		self.theMessageWindow = MessageWindow.MessageWindow()

		# -------------------------------------
		# creates Session
		# -------------------------------------
		self.theSession = ecell.Session.Session( ecell.ecs.Simulator() )
		self.theSession.setPrintMethod( self.theMessageWindow.printMessage )

		# -------------------------------------
		# creates LoggerWindow
		# -------------------------------------
		self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )

		# -------------------------------------
		# creates InterfaceWindow
		# -------------------------------------
		self.theInterfaceWindow = InterfaceWindow.InterfaceWindow( self )

		# -------------------------------------
		# creates PluginManager
		# -------------------------------------
		self.thePluginManager = OsogoPluginManager( self.theSession, 
		        self.theLoggerWindow, self.theInterfaceWindow, self.theMessageWindow )
		self.thePluginManager.loadAll()

		# -------------------------------------
		# creates PaletteManager
		# -------------------------------------
		self.thePaletteWindow = PaletteWindow.PaletteWindow()
		self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

		self.theEntryChecker = 0
		self.theStepperChecker = 0

		self.theUpdateInterval = 150
		self.theStepSize = 1
		self.theStepType = 0
		self.theRunningFlag = 0

		self.theHandlerMap = \
			{ 'load_rule_menu_activate'              : self.openRuleFileSelection ,
			'load_script_menu_activate'            : self.openScriptFileSelection ,
			'save_cell_state_menu_activate'        : self.saveCellStateToTheFile ,
			'save_cell_state_as_menu_activate'     : self.openSaveCellStateFileSelection ,
			'exit_menu_activate'                   : self.exit ,
			'message_window_menu_activate'         : self.toggleMessageWindow ,
			'interface_window_menu_activate'       : self.toggleInterfaceListWindow ,
			'palette_window_menu_activate'         : self.togglePaletteWindow ,
			'create_new_entry_list_menu_activate'  : self.toggleEntryList ,
			'create_new_logger_list_menu_activate' : self.toggleLoggerWindow ,
			'preferences_menu_activate'            : self.openPreferences ,
			'about_menu_activate'                  : self.openAbout ,
			'start_button_clicked'     : self.startSimulation ,
			'stop_button_clicked'      : self.stopSimulation ,
			'step_button_clicked'      : self.stepSimulation ,
			'input_step_size'          : self.setStepSize ,
			'step_sec_toggled'         : self.changeStepType ,
			'entrylist_togglebutton_toggled'     : self.toggleEntryList ,
			'logger_togglebutton_toggled'    : self.toggleLoggerWindow ,
			'palette_togglebutton_toggled'   : self.togglePaletteWindow ,
			'message_togglebutton_toggled'   : self.toggleMessageWindow ,
			'interface_togglebutton_toggled' : self.toggleInterfaceListWindow ,
		}
		self.addHandlers( self.theHandlerMap )


		# -------------------------------------
		# create Script File Selection 
		# -------------------------------------
		self.theScriptFileSelection = gtk.GtkFileSelection( 'Select Script File' )
		self.theScriptFileSelection.ok_button.connect('clicked', self.loadScript)
		self.theScriptFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theScriptFileSelection.complete( '*.' + self.ScriptFileExtension )

		# -------------------------------------
		# create Rule File Selection 
		# -------------------------------------
		self.theRuleFileSelection = gtk.GtkFileSelection( 'Select Rule File' )
		self.theRuleFileSelection.ok_button.connect('clicked', self.loadRule)
		self.theRuleFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theRuleFileSelection.complete( '*.' + self.RuleFileExtension )

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

	# end of __init__


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
		self['entry_list'].set_sensitive(0)
		self['palette_togglebutton'].set_sensitive(0)
		self['create_new_entry_list_menu'].set_sensitive(0)

	# end of setInitialWidgetStatus


	# ---------------------------------------------------------------
	# closeParentWindow
	#
	# obj: reference to widget like button....
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def closeParentWindow( self, obj ):

		aParentWindow = obj.get_parent_window()
		aParentWindow.hide()

	# end of closeParentWindow


	# ---------------------------------------------------------------
	# openRuleFileSelection
	#
	# obj: widget
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openRuleFileSelection( self, obj ) :

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
			self.theRuleFileSelection.hide()
			self.theSession.printMessage( 'loading rule file %s\n' % aFileName)
			aModelFile = open( aFileName )
			self.theSession.loadModel( aModelFile )
			aModelFile.close()
			self.theSession.theSimulator.initialize()
			self.update()
		except:
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)
		else:
			self.theStepperChecker = 1
			self['start_button'].set_sensitive(1)
			self['stop_button'].set_sensitive(1)
			self['step_button'].set_sensitive(1)
			self['entry_list'].set_sensitive(1)
			self['palette_togglebutton'].set_sensitive(1)
			self['create_new_entry_list_menu'].set_sensitive(1)
			self['load_rule_menu'].set_sensitive(0)
			self['load_script_menu'].set_sensitive(0)

	# end of loadRule


	# ---------------------------------------------------------------
	# openScriptFileSelection
	#
	# obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openScriptFileSelection( self, obj ) :

		self.theScriptFileSelection.show_all()
        
	# end of openScriptFileSelection


	# ---------------------------------------------------------------
	# loadScript
	#
	# obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def loadScript( self, obj ):

		try:
			aFileName = self.theScriptFileSelection.get_filename()
			self.theScriptFileSelection.hide()
			self.theSession.printMessage( 'loading script file %s\n' % aFileName )
			aGlobalNameMap = { 'theMainWindow' : self }
			execfile( aFileName, aGlobalNameMap )
			self.update()
		except:
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)
		else:
			self.theStepperChecker = 1
			self['start_button'].set_sensitive(1)
			self['stop_button'].set_sensitive(1)
			self['step_button'].set_sensitive(1)
			self['entry_list'].set_sensitive(1)
			self['palette_togglebutton'].set_sensitive(1)
			self['create_new_entry_list_menu'].set_sensitive(1)
			self['load_rule_menu'].set_sensitive(0)
			self['load_script_menu'].set_sensitive(0)

	# end of loadScript


	# ---------------------------------------------------------------
	# openSaveCellStateFileSelection
	#
	# obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openSaveCellStateFileSelection( self, obj ) :

		self.theSaveFileSelection.show_all()

	# end of openSaveCellStateFileSelection


	# ---------------------------------------------------------------
	# saveCellState ( NOT IMPLEMENTED ON 27/Jul/2002)
	#
	# obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def saveCellState ( self, obj ) :

		pass

	# end of saveCellState

	# ---------------------------------------------------------------
	# exit
	#
	# obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def exit( self, obj ):
		mainQuit()
        
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
				self.theLoggerWindow.update()

		except:
			import sys
			import traceback
			aErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(aErrorMessage)

		else:
			self.theRunningFlag = 0

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
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def changeStepType ( self, obj ):

		self.theStepType = 1 - self.theStepType

	# end of changeStepType


	# ---------------------------------------------------------------
	# setStepSize
	#
	# obj:  textfield
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setStepSize( self, obj ):

		aNumberString = obj.get_text()

		if len( aNumberString ):
			self.theStepSize = string.atof( aNumberString )
		else:
			self.theStepSize = 1

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
	# toggleEntryList
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleEntryList( self, button_obj ):

		#fix me: this part is Root System's bug.
		if self.theStepperChecker == 1 and self.theEntryChecker == 0:

			self.theEntryListWindow = EntryListWindow.EntryListWindow( self )
			self.theEntryChecker = 1

		if self.theStepperChecker == 0 and self.theEntryChecker == 0:
			self.theSession.printMessage( 'WARNING:need load model or script!!\n' )

		if self.theEntryChecker == 1:
			# ------------------------------------------------------
			# button is toggled to active 
			# ------------------------------------------------------
			if button_obj.get_active():

				# --------------------------------------------------
				# If instance of Entry Window Widget has destroyed,
				# creates new instance of Entry Window Widget.
				# --------------------------------------------------
				if ( self.theEntryListWindow.getExist() == 0 ):
					self.theEntryListWindow = EntryListWindow.EntryListWindow( self )
					#self.theEntryListWindow.openWindow()
					#self.theEntryListWindow.update()

				# --------------------------------------------------
				# If instance of Entry Window Widget has not destroyed,
				# calls show method of Entry Window Widget.
				# --------------------------------------------------
				else:
					self.theEntryListWindow['EntryListWindow'].show_all()

			# ------------------------------------------------------
			# button is toggled to non-active
			# ------------------------------------------------------
			else:

				# --------------------------------------------------
				# If instance of EntryList Window Widget has destroyed,
				# do nothing.
				# --------------------------------------------------
				if ( self.theEntryListWindow.getExist() == 0 ):
					pass

				# --------------------------------------------------
				# If instance of Entry Window Widget has not destroyed,
				# calls hide method of Entry Window Widget.
				# --------------------------------------------------
				else:
					self.theEntryListWindow['EntryListWindow'].hide()


	# end of toggleEntryList


	# ---------------------------------------------------------------
	# toggleLoggerWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindow( self, button_obj ):

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if button_obj.get_active() :

			# --------------------------------------------------
			# If instance of Logger Window Widget has destroyed,
			# creates new instance of Logger Window Widget.
			# --------------------------------------------------
			if ( self.theLoggerWindow.getExist() == 0 ):
				#self.theLoggerWindow.openWindow()
				self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
				self.theLoggerWindow.update()

			# --------------------------------------------------
			# If instance of Logger Window Widget has not destroyed,
			# calls show method of Logger Window Widget.
			# --------------------------------------------------
			else:
				self.theLoggerWindow['LoggerWindow'].show_all()

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


	# end of toggleLoggerWindow


	# ---------------------------------------------------------------
	# toggleMessageWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindow( self, button_obj ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if button_obj.get_active() :

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
				self.theMessageWindow['MessageWindow'].show_all()

		# ------------------------------------------------------
		# button is toggled to non-active
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

		self.thePaletteWindow = PaletteWindow.PaletteWindow()
		self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

	# end of initializePaletteWindow

	# ---------------------------------------------------------------
	# togglePaletteWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def togglePaletteWindow( self, button_obj ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if button_obj.get_active():

			if self.thePaletteWindow.getExist() == 0:
				self.initializePaletteWindow()
			else:
				pass	

			self.thePaletteWindow.show_all()

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:
			if self.thePaletteWindow.getExist() == 0:
				pass
			else:
				self.thePaletteWindow.hide()
        
	# end of togglePaletteWindow


	# ---------------------------------------------------------------
	# toggleInterfaceWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceListWindow( self, button_obj ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if button_obj.get_active() : 

			# --------------------------------------------------
			# If instance of Interface Window Widget has destroyed,
			# creates new instance of Interface Window Widget.
			# --------------------------------------------------
			if ( self.theInterfaceWindow.getExist() == 0 ):
				self.theInterfaceWindow.openWindow()
				#self.theInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
				self.theInterfaceWindow.update()

			# --------------------------------------------------
			# If instance of Interface Window Widget has not destroyed,
			# calls show method of Interface Window Widget.
			# --------------------------------------------------
			else:
				self.theInterfaceWindow['InterfaceWindow'].show_all()

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

	# end of toggleInterfaceListWindow


	# ---------------------------------------------------------------
	# saveCellStateToTheFile ( NOT IMPLEMENTED ON 27/Jul/2002)
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def saveCellStateToTheFile( self ):
		pass


# end of MainWindow

