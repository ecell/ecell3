#!/usr/bin/env python

from Window import *
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
    ]
DESCRIPTION = 'Osogo is a simulation session monitoring module for E-CELL SE Version 3'


class MainWindow(Window):

	def __init__( self ):
	
		Window.__init__( self )

		#### create Message Window ####
		self.theMessageWindow = MessageWindow.MessageWindow()
		self.theMessageWindowWindow = self.theMessageWindow[ 'message_window' ]

		self.theSession = ecell.Session.Session( ecell.ecs.Simulator() )
		self.theSession.setPrintMethod( self.theMessageWindow.printMessage )

		self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
		self.theLoggerWindowWindow = self.theLoggerWindow[ 'logger_window' ]

		self.theInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
		self.theInterfaceWindowWindow = self.theInterfaceWindow[ 'interface_window' ]
		#	 self.theInterfaceWindowWindow.hide()

		#self.thePluginManager = PluginManager( self.theSession, self.theLoggerWindow, self.theInterfaceWindow )
		#self.thePluginManager = OsogoPluginManager( self.theSession, self.theLoggerWindow, self.theInterfaceWindow )
		self.thePluginManager = OsogoPluginManager( self.theSession, self.theLoggerWindow, self.theInterfaceWindow, self.theMessageWindow )
		self.thePluginManager.loadAll()

		self.thePaletteWindow = PaletteWindow.PaletteWindow()
		self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

		#self.theEntryListWindow = EntryListWindow.EntryListWindow( self )
		#self.theEntryListWindowWindow = self.theEntryListWindow[ 'entry_list_window' ]

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


		#### create Script File Selection ####
		self.theScriptFileSelection = gtk.GtkFileSelection( 'Select Script File' )
		self.theScriptFileSelection.ok_button.connect('clicked', self.loadScript)
		self.theScriptFileSelection.cancel_button.connect('clicked', self.closeParentWindow)

		#### create Rule File Selection ####
		self.theRuleFileSelection = gtk.GtkFileSelection( 'Select Rule File' )
		self.theRuleFileSelection.ok_button.connect('clicked', self.loadRule)
		self.theRuleFileSelection.cancel_button.connect('clicked', self.closeParentWindow)

		#### create Save Cell State File Selection ####
		self.theSaveFileSelection = gtk.GtkFileSelection( 'Select Rule File for Saving' )
		self.theSaveFileSelection.ok_button.connect('clicked', self.saveCellState)
		self.theSaveFileSelection.cancel_button.connect('clicked', self.closeParentWindow)

		### initialize for run method ###
		self.theSession.theSimulator.setPendingEventChecker( gtk.events_pending )
		self.theSession.theSimulator.setEventHandler( gtk.mainiteration  )

		self['ecell_logo_toolbar'].set_style( GTK.TOOLBAR_ICONS )

	###### window operation ####
	def closeParentWindow( self, button_obj):
		aParentWindow = button_obj.get_parent_window()
		aParentWindow.hide()

	###### Load Rule ######
	def openRuleFileSelection( self, obj ) :
		self.theRuleFileSelection.show_all()

	def loadRule( self, button_obj ) :
		self.theStepperChecker = 1
		aFileName = self.theRuleFileSelection.get_filename()
		self.theRuleFileSelection.hide()
		self.theSession.printMessage( 'loading rule file %s\n' % aFileName)
		aModelFile = open( aFileName )
		self.theSession.loadModel( aModelFile )
		aModelFile.close()
		self.theSession.theSimulator.initialize()
		self.update()

	###### Load Script ######
	def openScriptFileSelection( self, obj ) :
		self.theScriptFileSelection.show_all()
        
	def loadScript( self, button_obj ):
		self.theStepperChecker = 1
		aFileName = self.theScriptFileSelection.get_filename()
		self.theScriptFileSelection.hide()
		self.theSession.printMessage( 'loading script file %s\n' % aFileName )
		aGlobalNameMap = { 'theMainWindow' : self }
		execfile( aFileName, aGlobalNameMap )
		self.update()

	###### Save Cell State As ######
	def openSaveCellStateFileSelection( self, obj ) :
		self.theSaveFileSelection.show_all()

	def saveCellState ( self, button_obj ) :
		pass

	###### Exit ######
	def exit( self, obj ):
		mainQuit()
        
	def startSimulation( self, a ) :

		# this can fail if the simulator is not ready
		self.theSession.theSimulator.initialize()

		self.theRunningFlag = 1
		self.theSession.printMessage( "Start\n" )
		self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.updateByTimeOut, 0)
		self.theLoggerWindow.update()
		self.theSession.run()
		self.removeTimeOut()

            
	def stopSimulation( self, a ) :
		if self.theRunningFlag:
			self.theRunningFlag = 0
			self.theSession.stop()
			self.theSession.printMessage( "Stop\n" )
			self.removeTimeOut()
			self.update()
			self.theLoggerWindow.update()


	def stepSimulation( self, a ) : 
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

            
	def changeStepType ( self, a ):
		self.theStepType = 1 - self.theStepType

	def setStepSize( self, obj ):
		aNumberString = obj.get_text()
		if len( aNumberString ):
			self.theStepSize = string.atof( aNumberString )
		else:
			self.theStepSize = 1

	def updateByTimeOut( self, a ):
		self.update()
		self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )

	def removeTimeOut( self ):
		gtk.timeout_remove( self.theTimer )

	def update( self ):
		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )
		self.thePluginManager.updateAllPluginWindow()
        
    
	def toggleEntryList( self, button_obj ):
		#fix me: this part is Root System's bug.
		if self.theStepperChecker == 1 and self.theEntryChecker == 0:
			self.theEntryListWindow = EntryListWindow.EntryListWindow( self )
			self.theEntryListWindowWindow = self.theEntryListWindow[ 'entry_list_window' ]
			self.theEntryChecker = 1
		if self.theStepperChecker == 0 and self.theEntryChecker == 0:
			self.theSession.printMessage( 'WARNING:need load model or script!!\n' )
		if self.theEntryChecker == 1:
			if button_obj.get_active() :
				self.theEntryListWindowWindow.show_all()
				self.theEntryListWindow.update()
			else :
				self.theEntryListWindowWindow.hide()



	def toggleLoggerWindow( self, button_obj ):

		if button_obj.get_active() :
			self.theLoggerWindowWindow.show_all()
			self.theLoggerWindow.update()
		else :
			self.theLoggerWindowWindow.hide()


	def toggleMessageWindow( self, button_obj ) :

		if button_obj.get_active() :
			self.theMessageWindowWindow.show_all()
		else :
			self.theMessageWindowWindow.hide()


	def openAbout( self, button_obj ):
		anAboutDialog = gnome.ui.GnomeAbout( NAME,
		                                     VERSION,
		                                     COPYRIGHT,
		                                     AUTHORLIST,
		                                     DESCRIPTION )
		anAboutDialog.set_title( 'about osogo' )
		anAboutDialog.show_all()


	def openPreferences( self, button_obj ):
		aPropertyBox = gnome.ui.GnomePropertyBox()
		aLabel = gtk.GtkLabel( 'NOT IMPLEMENTED YET' )
		aTabLabel = gtk.GtkLabel( 'warning' )
		aPropertyBox.append_page( aLabel, aTabLabel )

		aPropertyBox.show_all()



##########################################################################


	###### Toggle Palette Window ######
	def togglePaletteWindow( self, button_obj ) :
		if button_obj.get_active() :
			self.thePaletteWindow.show_all()
		else :
			self.thePaletteWindow.hide()
        
	def toggleInterfaceListWindow( self, button_obj ) :
		if button_obj.get_active() : 
			self.theInterfaceWindowWindow.show_all()
		else :
			self.theInterfaceWindowWindow.hide()

	#### these method is not supported in summer GUI project
	def saveCellStateToTheFile( self ) : pass
	#### these method is not supported in summer GUI project




