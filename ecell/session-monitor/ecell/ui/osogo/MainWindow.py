#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#


import gtk
import gobject
import string
import sys
import traceback
import os
import math
from time import time
from datetime import datetime

from ecell.ecssupport import *
from ecell.ecs_constants import *
import ecell.converter.sbml2eml
import ecell.eml

from OsogoWindow import OsogoWindow
from AboutSessionMonitor import AboutSessionMonitor
from MessageWindow import MessageWindow
from EntityListWindow import EntityListWindow
from LoggingPolicy import LoggingPolicy
from utils import *

import config

class SimulationButton:
    def __init__( self, aIconWidget, aLabelWidget ):
        # Image
        self.startImage = os.path.join( config.glade_dir, "icon_start.png" )
        self.stopImage = os.path.join( config.glade_dir, "icon_stop.png" )
        self.theIconWidget = aIconWidget
        self.theLabelWidget = aLabelWidget
        self.theCurrentState = None
        self.setCurrentState( False )

    def getCurrentImage( self ):
        return self.image

    def getCurrentState( self ):
        return self.theCurrentState

    def setCurrentState( self, aCurrentState ):
        self.theCurrentState = aCurrentState
        if self.theCurrentState:
            self.theIconWidget.set_from_file( self.stopImage )
            self.theLabelWidget.set_text( 'Stop' )
        else:
            self.theIconWidget.set_from_file( self.startImage )
            self.theLabelWidget.set_text( 'Start' )

class LogoAnimation:
    def __init__( self ):
        self.image = gtk.Image()
        self.image.set_from_file(
            os.path.join( config.glade_dir,  "ecell32.png" ) )
        self.image.show()

        aAnimPictureFileList = [
            'ecell32-%d.png' % seq for seq in range(1, 8)
        ] + [ 'ecell32.png' ];
        self.iconList = [
            os.path.join( config.glade_dir, filename )
                for filename in aAnimPictureFileList
            ];

        self.START_ROTATION = 0
        self.END_ROTATION = len( self.iconList) - 1

        self.__currentImage = 0
        self.__running = False

        self.extraCount = 0
        self.delay = 100
        self.theTimer = None

    def getImage( self ):
        return self.image

    def start( self ):
        self.__currentImage = 0
        self.__running = True

        self.animate()

    def stop( self ):
        self.__running = False
        
    def animate( self ):
        if ( self.__running ):
            if ( self.__currentImage == self.END_ROTATION ):
                if( self.extraCount == 5 ):
                    self.__currentImage = self.START_ROTATION
                    self.extraCount = 0
                else:
                    self.__currentImage = self.END_ROTATION - 1
                    self.extraCount += 1

            self.image.set_from_file( self.iconList[self.__currentImage] )
            self.__currentImage += 1
            self.theTimer = gobject.timeout_add( self.delay, LogoAnimation.animate, self )
        else:
            if ( self.__currentImage != self.END_ROTATION ):
                self.image.set_from_file( self.iconList[self.__currentImage] )
                self.__currentImage += 1
                self.theTimer = gobject.timeout_add( 60, LogoAnimation.animate, self )
            else:
                if self.theTimer != None:
                    gobject.source_remove( self.theTimer )
                    self.theTimer = None

    def isAnimating( self ):
        return self.__running

class Timer:
    def __init__( self, aCurrentTimeRetrieverDelegate, aTimerWidget ):
        self.theCurrentTimeRetrieverDelegate = aCurrentTimeRetrieverDelegate
        self.theCurrentTime = 0
        self.theCurrentRealTime = 0
        self.theTimerWidget = aTimerWidget
        self.theStartTime = 0
        self.theLastTime = 0
        self.theLastRealTime = 0
        self.theTimerWidget.set_text( str( 0 ) )
        self.theTimerWidget.set_property( 'xalign', 1 )

    def sync( self ):
        self.theCurrentTime = self.theCurrentTimeRetrieverDelegate()
        self.theCurrentRealTime = time()
        self.update()

    def reset( self ):
        if self.theStartTime == 0:
            self.theLastTime = self.theCurrentTime = 0
            self.theLastRealTime = self.theCurrentRealTime = self.theStartTime = 0
        else:
            self.theLastRealTime = \
                self.theCurrentRealTime = \
                self.theStartTime = time()
            self.theLastTime = \
                self.theCurrentTime = \
                self.theCurrentTimeRetrieverDelegate()
        self.update()

    def start( self ):
        self.theStartTime = -1
        self.reset()

    def stop( self ):
        self.update()
        self.theStartTime = 0

    def getCurrentSpeedRatio( self ):
        aRetVal = ( self.theCurrentTime - self.theLastTime ) / \
            ( self.theCurrentRealTime - self.theLastRealTime )
        self.theLastTime = self.theCurrentTime
        self.theLastRealTime = self.theCurrentRealTime
        return aRetVal

    def update( self ):
        if self.theStartTime != 0:
            aTime = self.theCurrentRealTime - self.theStartTime
            elapsed = datetime.utcfromtimestamp( aTime )
            self.theTimerWidget.set_text(
                "%d:%02d:%02d" % \
                ( elapsed.hour, elapsed.minute, elapsed.second ) )

class MainWindow( OsogoWindow ):
    def __init__( self ):
        # calls super class's constructor
        OsogoWindow.__init__( self )
        self.isLocked = False
        self.theMessageWindowVisible = False
        self.theEntityListWindowVisible = False
        self.timerVisible = False
        self.indicatorVisible = False

    def tryLock( self ):
        if self.isLocked:
            return False
        self.isLocked = True
        return True

    def unlock( self ):
        self.isLocked = False

    def initUI( self ):
        # calls superclass's method
        OsogoWindow.initUI( self )

        self.logoMovable = True
        self.theToolbarVisible = True
        self.theStatusbarVisible = True
                
        # create SimulationButton
        self.theSimulationButton = SimulationButton(
            self['simulation_button_icon'],
            self['simulation_button_label'])

        # create logo button
        self.theLogoAnimation = LogoAnimation()
        self['logo_animation'].add( self.theLogoAnimation.getImage() )

        # initialize time entry
        self['sec_step_entry'].set_property( 'xalign', 1 )
        self['time_entry'].set_property( 'xalign', 1 )

        # creates MessageWindow 
        self.theMessageWindow = MessageWindow()
        self.theMessageWindow.initUI()
        self.addChild( self.theMessageWindow, 'messagearea' )

        messageWindowSize = self.theMessageWindow.getActualSize()
        self.theMessageWindow['scrolledwindow1'].set_size_request(
            messageWindowSize[0], messageWindowSize[1] )
        self.setMessageWindowActive( True )

        # creates EntityListWindow 
        self.theEntityListWindow = self.theSession.createManagedWindow(
            EntityListWindow )
        self.addChild( self.theEntityListWindow, 'entitylistarea' )
        self.theEntityListWindow.initUI()
        self.setEntityListWindowActive( True )

        # ?
        self.theStepSizeOrSec = 1

        # initializes FileSelection reference
        self.theFileSelection = None

        # initialize Timer
        self.theTimerUpdateCount = 0
        self.theTimer = Timer(
            lambda: self.theSession.getCurrentTime(),
            self['timer_entry'] )
        self['timer_box'].hide()
        self.setTimerActive( False )

        # initialize Indicator
        self['indicator_box'].hide()
        self.setIndicatorActive( False )

        # append signal handlers
        self.addHandlers( {
            'exit_menu_activate': self.handleDeleteEvent,
            } )
        self.addHandlersAuto()
        self.update()

    def openFileSelection( self, aType, aTarget ) :
        """displays FileSelection 
        aType     ---  'Load'/'Save' (str)
        aTarget   ---  'Model'/'Script'/'SBML' (str)
        Return None
        [Note]:When FileSelection is already created, moves it to the top of desktop,
               like singleton pattern.
        """
        
        # When the FileSelection is already displayed, moves it to the top of desktop.
        if self.theFileSelection != None:
            self.theFileSelection.present()

        # When the FileSelection is not created yet, creates and displays it.
        else:

            # creates FileSelection
            self.theFileSelection = gtk.FileSelection()
            self.theFileSelection.connect(
                'delete_event', self.__deleteFileSelection )
            self.theFileSelection.cancel_button.connect(
                'clicked',
                self.__deleteFileSelection) 
            aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.glade_dir, 'ecell.png' ) )
            aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.glade_dir, 'ecell32.png' ) )
            self.theFileSelection.set_icon_list(aPixbuf16, aPixbuf32)

            # when 'Load Model' is selected
            if aType == 'Load' and aTarget == 'Model':
                self.theFileSelection.ok_button.connect(
                    'clicked', self.__loadData, aTarget)
                self.theFileSelection.complete( '*.' + MODEL_FILE_EXTENSION )
                self.theFileSelection.set_title(
                    "Select %s File (%s)" % ( aTarget,MODEL_FILE_EXTENSION ) )
            # when 'Load Script' is selected
            elif aType == 'Load' and aTarget == 'Script':
                self.theFileSelection.ok_button.connect(
                    'clicked', self.__loadData, aTarget )
                self.theFileSelection.complete( '*.' + SCRIPT_FILE_EXTENSION )
                self.theFileSelection.set_title(
                    "Select %s File (%s)" % (aTarget,SCRIPT_FILE_EXTENSION) )
            # when 'Save Model' is selected
            elif aType == 'Save' and aTarget == 'Model':
                self.theFileSelection.ok_button.connect(
                    'clicked', self.__saveModel )
                self.theFileSelection.complete( '*.' + MODEL_FILE_EXTENSION )
                self.theFileSelection.set_title(
                    "Select %s File (%s)" % ( aTarget, MODEL_FILE_EXTENSION ) )
            # when 'Import SBML' is selected
            elif aType == 'Load' and aTarget == 'SBML':
                self.theFileSelection.ok_button.connect(
                    'clicked', self.__loadData, aTarget)
                self.theFileSelection.complete( '*.xml' )
                self.theFileSelection.set_title(
                    "Select %s File (%s)" % ( aTarget, '.xml' ) )
            # when 'Save Model' is selected
            elif aType == 'Save' and aTarget == 'SBML':
                self.theFileSelection.ok_button.connect(
                    'clicked', self.__exportSBML)
                self.theFileSelection.complete( '*.xml' )
                self.theFileSelection.set_title(
                    "Select %s File (%s)" % ( aTarget, ".xml" ) )
            else:
                raise "(%s,%s) does not match." %(aType,aTarget)

            # displays the created FileSelection
            self.theFileSelection.show_all()

    def destroy( self ):
        """ restores message method and closes window """
        OsogoWindow.destroy( self )
        self.theSession.terminate()

    def getStepType( self ):
        """ returns state of sec radiobutton
            True: seconds
            False: steps
        """
        return self['sec_radiobutton'].get_active()

    def setStepType( self, aState ):
        """ sets Step Type radiobutton state 
            values for aState
            True : seconds
            False : step            
        """
        if aState == True :
            self['sec_radiobutton'].set_active( True )
        if aState == False:
            self['sec_radiobutton'].set_active( False )

    def getStepSize( self ):
        """ returns user or script specifid step size """
        return self.theStepSizeOrSec

    def setStepSize( self, num ):
        """ sets Stepsize entry box to num """
        assert num > 0
        self.theStepSizeOrSec = num
        self['sec_step_entry'].set_text( str( num ) )

    def update( self ):
        """updates this window 
        Returns None
        """
        if not self.exists():
            return None
        self.updateTimer()

        # updates status of menu and button 
        self.updateMenuAndButtons()

        self.theEntityListWindow.update()
        self.theEntityListWindow.theQueue.applyFullPNList()
    
    def handleSessionEvent( self, event ):
        if event.type == 'simulation_stopped':
            self.theLogoAnimation.stop()
            self.theSimulationButton.setCurrentState( False )
            self.updateTimer()
        elif event.type == 'simulation_started':
            self.theTimer.reset()
            self.theTimer.start()
            if self.logoMovable:
                self.theLogoAnimation.start()
            self.theSimulationButton.setCurrentState( True )
        elif event.type == 'simulation_updated':
            self.updateTimer()
            self['time_entry'].set_text( str( event.simulationTime ) )
            self['sec_step_entry'].set_text( str( self.theStepSizeOrSec ) )
        elif event.type == 'window_shown':
            self.setToolbarButtonState( event.target_name, True )
        elif event.type == 'window_hidden':
            self.setToolbarButtonState( event.target_name, False )
        elif event.type == 'model_loaded':
            self.theEntityListWindow.update()
            self.setMenuAndButtonState( True )
        elif event.type == 'model_unloaded':
            self.setMenuAndButtonState( False )
        elif event.type == 'message':
            self.theMessageWindow.printMessage( event.content )
        elif event.type == 'fundamental_window_created':
            self.setToolbarButtonState( event.window_name, True )
        elif event.type == 'fundamental_window_destroyed':
            self.setToolbarButtonState( event.window_name, False )

    def setToolbarButtonState( self, target_name, flag ):
        if target_name == 'BoardWindow':
            self['board_window_menu'].set_active( flag )
            self['board_button'].get_child().set_active( flag )
        elif target_name == 'LoggerWindow':
            self['logger_window_menu'].set_active( flag )
            self['logger_button'].get_child().set_active( flag )
        elif target_name == 'InterfaceWindow':
            self['interface_window_menu'].set_active( flag )
            self['interface_button'].get_child().set_active( flag )
        elif target_name == 'StepperWindow':
            self['stepper_window_menu'].set_active( flag )
            self['stepper_button'].get_child().set_active( flag )
        elif target_name == 'MessageWindow':
            self['message_window_menu'].set_active( flag )
            self['message_button'].get_child().set_active( flag )
        elif target_name == 'EntityListWindow':
            self['entitylist_window_menu'].set_active( flag )
            self['entitylist_button'].get_child().set_active( flag )

    def updateMenuAndButtons( self ):
        """
        DEPRECATED
        sets initial widgets status
        aDataLoadedStatus  -- the status of loading data
                 (True:Model or Script is loaded / False:Not loaded)
        Returns None
        """
        self.setMenuAndButtonState( self.theSession.isModelLoaded() )

    def setMenuAndButtonState( self, aDataLoadedStatus ):
        # toolbar
        self['simulation_button'].set_sensitive(aDataLoadedStatus)
        self['step_button'].set_sensitive(aDataLoadedStatus)
        self['timer_clear_button'].set_sensitive(aDataLoadedStatus)
        self['load_model_button'].set_sensitive(not aDataLoadedStatus)
        self['load_script_button'].set_sensitive(not aDataLoadedStatus)
        self['save_model_button'].set_sensitive(aDataLoadedStatus)
        self['logger_button'].set_sensitive(aDataLoadedStatus)
        self['stepper_button'].set_sensitive(aDataLoadedStatus)
        self['interface_button'].set_sensitive(aDataLoadedStatus)
        self['board_button'].set_sensitive(aDataLoadedStatus)
        self['indicator_button'].set_sensitive(aDataLoadedStatus)
        self['timer_button'].set_sensitive(aDataLoadedStatus)

        # file menu
        self['load_model_menu'].set_sensitive(not aDataLoadedStatus)
        self['load_script_menu'].set_sensitive(not aDataLoadedStatus)
        self['save_model_menu'].set_sensitive(aDataLoadedStatus)
        self['import_sbml'].set_sensitive(not aDataLoadedStatus)
        self['export_sbml'].set_sensitive(aDataLoadedStatus)

        # window menu
        self['logger_window_menu'].set_sensitive(aDataLoadedStatus)
        self['stepper_window_menu'].set_sensitive(aDataLoadedStatus)
        self['interface_window_menu'].set_sensitive(aDataLoadedStatus)
        self['board_window_menu'].set_sensitive(aDataLoadedStatus)
        self['entity_list_menu'].set_sensitive(aDataLoadedStatus)
        self['save_model_menu'].set_sensitive(aDataLoadedStatus)

        # preferences menu
        self['logging_policy'].set_sensitive(aDataLoadedStatus)
        self['run_speed_indicator'].set_sensitive(aDataLoadedStatus)
        self['timer_menu'].set_sensitive(aDataLoadedStatus)
        self['logo_animation_menu'].set_sensitive(aDataLoadedStatus)

    def updateTimer( self ):
        if self.theSession.isRunning():
            self.theTimer.sync()
            self.theTimerUpdateCount += 1
            if self.theTimerUpdateCount == 25:
                self['run_speed_label'].set_text(
                    str( round( self.theTimer.getCurrentSpeedRatio(), 5 ) ) )
                self.theTimerUpdateCount = 0

    def setIndicatorActive( self, isActive ): 
        self['run_speed_indicator'].set_active(isActive)
        self['indicator_button'].get_child().set_active(isActive)
        self.indicatorVisible = isActive
        if isActive:
            self['indicator_box'].show()
        else:
            self['indicator_box'].hide()

    def setTimerActive( self, isActive ): 
        self['timer_menu'].set_active(isActive)
        self['timer_button'].get_child().set_active(isActive)
        self.timerVisible = isActive
        if isActive:
            self['timer_box'].show()                
            self.updateTimer()
        else:
            self['timer_box'].hide()

    def setLogoMovable( self, state ):
        if state:
            self.logoMovable = False
            if self.theLogoAnimation.isAnimating():
                self.theLogoAnimation.stop()
        else:
            self.logoMovable = True
            if not self.theLogoAnimation.isAnimating():
                self.theLogoAnimation.start()
 
    def setMessageWindowActive( self, state ):
        if state:
            self[ 'message_handlebox' ].show()
            self.theMessageWindowVisible = True
            self.setToolbarButtonState( 'MessageWindow', True )
            self.__resizeVertically(
                ( self.theEntityListWindowVisible and \
                    self['entitylistarea'].get_allocation()[3] or \
                    0 ) + self.theMessageWindow.getActualSize()[1]
                )
        else:
            self[ 'message_handlebox' ].hide()
            self.theMessageWindowVisible = False
            self.setToolbarButtonState( 'MessageWindow', False )
            self.__resizeVertically(
                self.theEntityListWindowVisible and \
                    self['entitylistarea'].get_allocation()[3] or \
                    0
                )

    def setEntityListWindowActive( self, state ):
        if state:
            self['entitylistarea'].show()
            self.theEntityListWindowVisible = True
            self.setToolbarButtonState( 'EntityListWindow', True )
            self.__resizeVertically( self['entitylistarea'].get_allocation()[3] )
            self.__resizeVertically(
                ( self.theMessageWindowVisible and \
                    self.theMessageWindow.getActualSize()[1] or \
                    0 ) + self['entitylistarea'].get_allocation()[3]
                )
        else:
            self['entitylistarea'].hide()
            self.theEntityListWindowVisible = False
            self.setToolbarButtonState( 'EntityListWindow', False )
            self.__resizeVertically(
                ( self.theMessageWindowVisible and \
                    self.theMessageWindow.getActualSize()[1] or \
                    0 )
                )

    def setToolbarVisibility( self, state ):
        if state:
            self['toolbar1_handlebox'].show()
            self['toolbar2_handlebox'].show()
            self.theToolbarVisible = True
        else:
            self['toolbar1_handlebox'].hide()
            self['toolbar2_handlebox'].hide()
            self.theToolbarVisible = False
            if self.theMessageWindowVisible == False and \
               self.theEntityListWindowVisible == False:
                   self.__resizeVertically( 0 )

    def setStatusbarVisibility( self, state ):
        # show Statusbar
        if state:
            self['statusbar'].show()
            self.theStatusbarVisible = True
        else:
            self['statusbar'].hide()
            self.theStatusbarVisible = False
            if self.theMessageWindowVisible == False and \
               self.theEntityListWindowVisible == False:
                   self.__resizeVertically( 0 )

    def showAboutSessionMonitor( self ):
        aWindow = AboutSessionMonitor()
        aWindow.run()
        aWindow.destroy()

    def show( self ):
        OsogoWindow.show( self )
        if not self.timerVisible:
            self['timer_box'].hide()
        if not self.indicatorVisible:
            self['indicator_box'].hide()
        if not self.theMessageWindowVisible:
            self['message_handlebox'].hide()
        if not self.theEntityListWindowVisible:
            self['entitylistarea']. hide()

    def __resizeVertically( self, height ): #gets entitylistarea or messagebox height
        """__resizeVertically
        Return None
        """

        # gets fix components height
        menu_height = self['menubar_handlebox'].get_child_requisition()[1]
        toolbar_height = \
            self['toolbar1_handlebox'].get_child_requisition()[1] + \
            self['toolbar2_handlebox'].get_child_requisition()[1]

        # gets window_width
        window_width=self['MainWindow'].get_size()[0]

        # resizes
        window_height=menu_height+toolbar_height+height
        self['MainWindow'].resize(window_width,window_height)
    
    def doInitiateLoadModel( self ) :
        self.openFileSelection('Load','Model')

    def doInitiateLoadScript( self ):
        self.openFileSelection('Load','Script')

    def doInitiateSaveModel( self ):
        self.openFileSelection('Save','Model')

    def doInitiateImportSBML( self ):
        self.openFileSelection('Load','SBML')

    def doInitiateExportSBML( self ):
        self.openFileSelection('Save','SBML')

    def __deleteFileSelection( self, *arg ):
        """deletes FileSelection
        Return None
        """

        # deletes the reference to FileSelection
        if self.theFileSelection != None:
            self.theFileSelection.destroy()
            self.theFileSelection = None

    def __loadData( self, w, aFileType ) :
        """loads model or script file
        arg[0]    ---   ok button of FileSelection
        arg[1]    ---   'Model'/'Script' (str)
        Return None
        """
        aFileName = self.theFileSelection.get_filename()
        if os.path.isfile( aFileName ):
            pass
        else:
            aMessage = ' Error ! No such file. \n[%s]' %aFileName
            self.theSession.message(aMessage)
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            self.theFileSelection.present()
            return None

        self.__deleteFileSelection()
        self.theSession.message('Loading %s file %s\n' %(aFileType, aFileName) )

        try:
            if aFileType == 'Model':
                self.theSession.loadModel( aFileName )
            elif aFileType == 'SBML':
                try:
                    io = open( aFileName, 'r' )
                except IOError:
                    self.theSession.message("Failed to load %s" % aFileName)
                    return
                aEMLString = ecell.converter.sbml2eml.convert( io.read() )
                self.theSession.loadModel( ecell.eml.Eml( aEMLString ) )
            elif aFileType == 'Script':
                self.theSession.loadScript( aFileName )
        except:
            # set load command not to be operated 
            self['load_model_button'].set_sensitive(0)
            self['load_script_button'].set_sensitive(0)
            self['load_model_menu'].set_sensitive(0)
            self['load_script_menu'].set_sensitive(0)

            # expants message window, when it is folded.
            if self.exists():
                if ( self['message_button'].get_child() ).get_active() == False:
                    ( self['message_button'].get_child() ).set_active(True)
            # displays confirm window
            aMessage = 'Can\'t load [%s]\nSee MessageWindow for details.' %aFileName
            showPopupMessage( OK_MODE, aMessage, 'Error' )

            # displays message on MessageWindow
            aMessage = 'Can\'t load [%s]' %aFileName
            self.theSession.message(aMessage)
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type, sys.exc_value, \
                    sys.exc_traceback), '\n' )
            self.theSession.message(anErrorMessage)

    def __saveModel( self, *arg ) :
        # gets file name
        aFileName = self.theFileSelection.get_filename()

        # when the file already exists
        if os.path.isfile( aFileName ):

            # displays confirm window
            aMessage = 'Would you like to replace the existing file? \n[%s]' %aFileName
            if showPopupMessage(
                OKCANCEL_MODE, aMessage,
                'Question' ) != OK_PRESSED:
                return None

        # when ok is pressed, overwrites it.
        # deletes FileSelection
        self.__deleteFileSelection()

        try:
            # displays save message
            self.theSession.message('Save model file %s\n' % aFileName)                        
            # saves Model
            self.theSession.saveModel( aFileName )

        except:
            # expants message window, when it is folded.
            if ( self['message_button'].get_child() ).get_active() == False:
                ( self['message_button'].get_child() ).set_active(True)

            # displays confirm window
            aMessage = 'Can\'t save [%s]\nSee MessageWindow for details.' %aFileName
            aDialog = showPopupMessage( OK_MODE, aMessage, 'Error' )
            # displays error message of MessageWindow
            self.theSession.message('Can\'t save [%s]' %aFileName)
            anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
            self.theSession.message(anErrorMessage)
        # updates
        self.theSession.updateUI()

    def __exportSBML( self, *arg ) :
        # gets file name
        aFileName = self.theFileSelection.get_filename()

        # when the file already exists
        if os.path.isfile( aFileName ):
            # displays confirm window
            aMessage = 'Would you like to replace the existing file? \n[%s]' %aFileName
            if showPopupMessage(
                OKCANCEL_MODE,
                aMessage, 'Question' ) != OK_PRESSED:
                return None

        # when ok is pressed, overwrites it.
        # deletes FileSelection
        self.__deleteFileSelection()

        try:
            # displays save message
            self.theSession.message('Export SBML file %s\n' % aFileName)                        
            # saves Model
            self.theSession.exportSBML( aFileName )

        except:
            # expants message window, when it is folded.
            if not self['message_button'].get_child().get_active():
                self['message_button'].get_child().set_active( True )

            # displays confirm window
            aMessage = "Can't save [%s]\nSee MessageWindow for details." % \
                       aFileName
            showPopupMessage( OK_MODE, aMessage, 'Error' )

            # displays error message of MessageWindow
            self.theSession.message("Can't export [%s]" % aFileName)
            anErrorMessage = string.join(
                traceback.format_exception(
                    sys.exc_type, sys.exc_value, sys.exc_traceback), '\n' )
            self.theSession.message(anErrorMessage)

        # updates
        self.theSession.updateFundamentalWindows()

    def checkWidgetState( self, widget_name ):
        if widget_name == 'simulation_button':
            return not self.theSimulationButton.getCurrentState()

    def handleDeleteEvent( self, *arg ):
        # stop simulation temporarily
        sessionHasBeenRunning = False
        if self.theSession.isRunning():
            self.theSession.stop()
            sessionHasBeenRunning = True

        # If there is no logger data, exit this program.
        if len( self.theSession.getLoggerList() ) != 0:
            # Popup confirm window, and check user request
            if showPopupMessage(
                OKCANCEL_MODE,
                'Are you sure to quit the application?',
                'Question'
                ) != OK_PRESSED:
                if sessionHasBeenRunning:
                    self.theSession.run()
                return False

        self.destroy()
        return True

    def doToggleSimulation( self, state ) :
        """handles simulation
        arg[0]  ---  simulation button (gtk.Button)
        Returns None
        """
        if state:
            self.theSession.run()
        else:
            self.theSession.stop()
 
    def doStepSimulation( self ) : 
        """steps simulation
        arg[0]  ---  stop button (gtk.Button)
        Returns None
        if step measure is second, then Session.run()
        if step measure is step than Session.step ()
        """
        if self.getStepType():
            self.theSession.run( self.getStepSize() )
        else:
            self.theSession.step( self.getStepSize() )

<<<<<<< .mine
    def doSetStepSizeOrSec( self, text ):
=======
    def doInputStepSizeOrSec( self ):
>>>>>>> .r3007
        # gets the inputerd characters from the GtkEntry. 
        aNewValue = string.strip( text )
        hasErrorOccurred = False

        try:
            # converts string to float
            aNewValue = string.atof( aNewValue )
        except ValueError:
            # displays a Confirm Window.
            aMessage = "\"%s\" is not numerical value."  % aNewValue
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            hasErrorOccurred = True
        # when string can be converted to float
        else:
            # validate the input
            aMessage = ""
            if aNewValue <= 0:
                aMessage += "Running period must be greater than 0.\n"
            # when 'step' is selected.
            if self['step_radiobutton'].get_active() == True:
                # step must be integer 
                if int( aNewValue ) != aNewValue:
                    aMessage += "Number of steps must be an integer.\n"
                aNewValue = int( aNewValue )
            if len(aMessage) > 0:
                showPopupMessage( OK_MODE, aMessage, 'Error' )
        # revert the change
        if hasErrorOccurred:
            self['sec_step_entry'].set_text( str( self.theStepSizeOrSec ) )
        else:
            self.theStepSizeOrSec = aNewValue

    def openLogPolicyWindow( self, aLogPolicy, aTitle = None ):
        """ pops up a modal dialog window
            with aTitle (str) as its title
            and displaying loggingpolicy
            and with an OK and a Cancel button
            users can set logging policy
            returns:
            logging policy if OK is pressed
            None if cancel is pressed
        """
        aLogPolicyWindow = LoggingPolicy()
        aLogPolicyWindow.setLoggingPolicy( aLogPolicy )
        if aTitle != None:
            assert aLogPolicyWindow.setTitle( aTitle )
        aLogPolicyWindow.initUI()
        aLogPolicyWindow.show()
        return aLogPolicyWindow.getResult()

    def doOpenLoggingPolicyDialog( self ):
        """
        signal handler to logging policy is called
        """
        # get default logging policy
        aLogPolicy = self.theSession.getLogPolicyParameters()
        
        # open logpolicy window
        newLogPolicy = self.openLogPolicyWindow(
            aLogPolicy, "Set default log policy" )
        if newLogPolicy != None:
            # save logpolicy
            self.theSession.setLogPolicyParameters( newLogPolicy )

    def doToggleToolbar( self, state ):
        self.setToolbarVisibility( state )

    def doToggleStatusbar( self, state ):
        self.setStatusbarVisibility( state )

    def doToggleIndicator( self, state ):
        self.setIndicatorActive( state )

    def doToggleTimer( self, state ):
        self.setTimerActive( state )

    def doToggleLoggerWindow( self, state ):
        self.theSession.setFundamentalWindowVisibility( 'LoggerWindow', state )

    def doToggleStepperWindow( self, state ):
        self.theSession.setFundamentalWindowVisibility( 'StepperWindow', state )

    def doToggleInterfaceWindow( self, state ):
        self.theSession.setFundamentalWindowVisibility( 'InterfaceWindow', state )

    def doToggleBoardWindow( self, state ):
        self.theSession.setFundamentalWindowVisibility( 'BoardWindow', state )

    def doToggleMessagePane( self, state ) :
        """expands or folds MessageWindow
        arg[0]   ---  self['message_button'] or self['message_window_menu']
        Returns None
        """
        self.setMessageWindowActive( state )

    def doToggleEntityListPane( self, state ):
        self.setEntityListWindowActive( state )

    def doDisplayAbout( self ):
        self.showAboutSessionMonitor()

    def doClearTimer( self ):
        self.theTimer.reset()

    def doToggleLogoAnimation( self, state ):
        self.setLogoMovable( True )

    def getStatusBar( self ):
        return self['statusbar']

    def doCreateEntityListWindow( self ):
        self.theSession.openManagedWindow( EntityListWindow )
