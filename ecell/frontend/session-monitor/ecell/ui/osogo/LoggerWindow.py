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
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

#from Window import *

import gobject
import gtk
from numpy import *

from os import *

import ecell.util as util
from OsogoWindow import OsogoWindow
from ecell.DataFileManager import DataFileManager
from ecell.ECDDataFile import ECDDataFile
from LoggingPolicy import *

class LoggerWindow( OsogoWindow ):
    def __init__( self ): 
        """
        constructor
        aSession : the reference of session
        aSession : the reference of Session
        """
        OsogoWindow.__init__( self )
        self.theDefaultSaveDirectory = 'Data' # XXX: avoid hardcoding!!
        self.thePopupMenu = None
        self.theList = []
        self.theSaveDirectorySelection = None

    def initUI( self ):
        OsogoWindow.initUI( self )

        self.thePopupMenu = PopupMenu( self )
        # Creates save file selection
        self.theSaveDirectorySelection = gtk.FileSelection( 'Select File' )
        self.theSaveDirectorySelection.ok_button.connect('clicked', self.changeSaveDirectory)
        self.theSaveDirectorySelection.cancel_button.connect('clicked', self.closeParentWindow)

        self.theEntryList = self['loggerWindow_clist']

        aListStore = gtk.ListStore( gobject.TYPE_STRING,\
                                    gobject.TYPE_STRING,\
                                    gobject.TYPE_STRING )


        self.theEntryList.set_model( aListStore )
        column=gtk.TreeViewColumn('FullPN',gtk.CellRendererText(),text=0)
        column.set_resizable(True)
        self.theEntryList.append_column(column)
        column=gtk.TreeViewColumn('Start',gtk.CellRendererText(),text=1)
        column.set_resizable(True)
        self.theEntryList.append_column(column)
        column=gtk.TreeViewColumn('End',gtk.CellRendererText(),text=2)
        column.set_resizable(True)
        self.theEntryList.append_column(column)
        self.theEntryList.get_selection().set_mode(gtk.SELECTION_MULTIPLE)
        # reset
        self.resetAllValues()
        # Adds handlers
        self.addHandlers(
            {
                # bottom basic bottons (save,close,reset)
                'on_save_button_clicked' : self.saveData, 
                'on_close_button_clicked' : self.handleDeleteEvent, 
                'on_reset_button_clicked' : self.resetAllValues, 

                # save_directory 
                'on_directory_button_clicked' : self.spawnSaveDirectorySelection,

                # data_interval
                'on_datainterval_checkbox_toggled' : self.updateDataInterval,

                # time
                'on_time_checkbox_toggled' : self.updateStartEnd,

                # popup
                'button_press_event'  : self.popupMenu
                }            
            )

    def destroy( self ):
        self.theEntryList = None
        OsogoWindow.destroy( self )

    def resetAllValues( self, obj=None ):
        """
        Resets all value
        return -> None
        """
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

    def closeParentWindow( self, obj ):
        """
        Closes parent window
        return -> None
        """
        aParentWindow = self.theSaveDirectorySelection.cancel_button.get_parent_window()
        aParentWindow.hide()

    def spawnSaveDirectorySelection( self, obj ):
        """
        Spawns
        return -> None
        """
        self.theSaveDirectorySelection.show_all()

    def changeSaveDirectory( self, obj ):
        """
        If directory_button is clicked, this method is called.
        return -> None
        """
        aSaveDirectory = self.theSaveDirectorySelection.get_filename()
        self['directory_entry'].set_text(aSaveDirectory)
        self.theSaveDirectorySelection.hide()

    def updateDataInterval( self, obj=None ):
        """
        If datainterval_checkbox is toggled, this method is called.
        return -> None
        """
        if self['datainterval_checkbox'].get_active():
            self['datainterval_spinbutton'].set_sensitive(0)
        else:
            self['datainterval_spinbutton'].set_sensitive(1)

    def updateStartEnd( self, obj=None ):
        """
        If time_checkbox is toggled, this method is called.
        return -> None
        """
        if self['time_checkbox'].get_active():
            self['start_spinbutton'].set_sensitive(0)
            self['end_spinbutton'].set_sensitive(0)
        else:
            self['start_spinbutton'].set_sensitive(1)
            self['end_spinbutton'].set_sensitive(1)

    def saveData( self, obj ):
        """
        If save_button is clicked, then this method is called.
        return -> None
        """
        aGeneralSaveErrorMessage="couldn't save files."

        # [0] clean up statusbar
        self["statusbar"].pop(1)

        # [1] checks status and value of each widget

        # [1-1] At least, one data must be selected.
        # If no list is selected, exit this method.
        if len(self.getSelectedPropertyNames()) == 0:
            self["statusbar"].push( 1, 'Select some data.' )
            anErrorMessage = 'No data is selected.'
            showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
            return None

        # [1-2] interval must be > 0
        # If interval is 0, exit this method.
        anInterval = -1
        if not self['datainterval_checkbox'].get_active():
            anInterval = self['datainterval_spinbutton'].get_value()
            if anInterval==0:
                self["statusbar"].push(1,'Set interval > 0.')
                anErrorMessage = 'Interval must be > 0.'
                showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
                return None


        # [1-3] Now binary type is not supported by Logger.
        # If binary type is selected, exit this method.
        aType = self["datatype_combo"].get_text()

        if aType == 'ecd':
            pass
        elif aType == 'binary':
            self["statusbar"].push(1,'Select ecd type.')
            anErrorMessage = "Sorry, binary format will be supported in the future version."
            showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
            return None

        # [1-4] start < end
        # If start >= end, exit this method.
        aStartTime=-1
        anEndTime=-1
        if not self['time_checkbox'].get_active():
            aStartTime = self['start_spinbutton'].get_value()
            anEndTime = self['end_spinbutton'].get_value()

            if aStartTime >= anEndTime:
                self["statusbar"].push(1,'Set start time < end time.')
                anErrorMessage = 'Start time must be < end time.!\n'
                showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
                return None

        # [2] Creates Data directory.
        aSaveDirectory = self['directory_entry'].get_text()

        # If same directory exists.
        if os.path.isdir(aSaveDirectory):
            aConfirmMessage = "%s directory already exist.\n Would you like to override it?"%aSaveDirectory
            if showPopupMessage( OKCANCEL_MODE, aConfirmMessage ) != OK_PRESSED:
                self["statusbar"].push(1, 'Saving log data was cancelled.')
                return None
        else:
            try:
                os.mkdir(aSaveDirectory)
                self["statusbar"].push(1,'Set start time < end time.')
            except:
                anErrorMessage = "Couldn't create %s!" % aSaveDirectory
                self["statusbar"].push(1, anErrorMessage )
                showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
                return None
            else:
                self["statusbar"].push( 1, '%s was created.' % aSaveDirectory )

        # [3] Execute saving.
        try:
            self.theSession.saveLoggerData(
                self.getSelectedPropertyNames(),
                aSaveDirectory,
                aStartTime, anEndTime,
                anInterval )
        except e:
            anErrorMessage =  "Failed to save the specified log to %s (reason: %s)" % ( aSaveDirectory, e.message )
            self["statusbar"].push( 1, anErrorMessage )
            showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
            return None
        
        aSuccessMessage = "All the selected files have been saved successfully." 
        self["statusbar"].push( 1 , aSuccessMessage )

    def getSelectedPropertyNames( self ):
        """
        Gets the selected PropertyName
        return -> selected propertyname list
        """
        aSelection = self[ 'loggerWindow_clist' ].get_selection()
        retval = []
        aSelection.selected_foreach(
            lambda aModel, aPath, anIter: retval.append(
                identifiers.FullPN( aModel.get_value( anIter, 0 ) )) )
        return retval

    def update( self ):
        """
        Updates
        return -> None
        """
        if not self.exists():
            return None

        self.theFullPNList = self.theSession.getLoggedPNList()
        self.theList = []

        for aFullPN in self.theFullPNList :
            aLoggerStub = self.theSession.createLogger( aFullPN )
            start = str( aLoggerStub.getStartTime() )
            if self.theSession.theRunningFlag:
                end = 'running'
            else:
                end = str( aLoggerStub.getEndTime() )
            aList = [ aFullPN, start, end ]
            self.theList.append( aList )
        aModel = self.theEntryList.get_model()
        aModel.clear()
        for aValue in self.theList:
            anIter = aModel.append()
            aModel.set(
                anIter,
                0, aValue[0],
                1, str(aValue[1]),
                2, str(aValue[2])
                )

    def handleSessionEvent( self, event ):
        if event.type == 'logger_created' or \
           event.type == 'logger_deleted':
            self.update()

    def popupMenu( self, aWidget, anEvent ):
        """
        popupMenu
          - show popup menu
        
        aWidget         : widget
        anEvent          : an event
        return -> None
        This method is throwable exception.
        """
        if len(self.getSelectedPropertyNames())!=0:
            if anEvent.button == 3:
                self.thePopupMenu.popup( None, None, None, 1, 0 )
        return False

    def editPolicy( self, *args ):
        if len(self.aSelectedPropertyNameList) == 1:
        # get loggerpolicy
            aLoggerStub = self.theSession.createLoggerStub( self.aSelectedPropertyNameList[0] )
            aLogPolicy = aLoggerStub.getLoggerPolicy()
        else:
            aLogPolicy = [0,0,0,0]
        newLogPolicy = self.theSession.openLogPolicyWindow( aLogPolicy, "Set log policy for selected loggers" )
        if newLogPolicy == None:
            return
        for aFullPN in self.aSelectedPropertyNameList:
            aLoggerStub = self.theSession.createLoggerStub( aFullPN )
            aLoggerStub.setLoggerPolicy( newLogPolicy )

    def saveDataFile( self, aFullPN, aDirectory = None, anInterval = None, aStartTime = None, anEndTime = None, fileType = 'ecd' ):
        """
        saves FullPN logger data with given parameters
        if parameters are not given, function defaults to window settings
        for parameters with a given value it sets appropriate window settings
        and saves FullPN into Datafile
        """
        #find FullPM in list
        anIter = self.theEntryList.get_model().get_iter_first()
        while True:
            if anIter == None:
                return None
            selectedFullPN = identifiers.FullPN(
                self.theEntryList.get_model().get_value( anIter, 0 ) )
            if selectedFullPN == aFullPN:
                aPath = self.theEntryList.get_model().get_path ( anIter )
                self.theEntryList.set_cursor( aPath, None, False )
                break
            anIter = self.theEntryList.get_model().iter_next( anIter )
        if aDirectory != None:
            self.setDirectory( aDirectory )
        
        if anInterval != None:
            self.setInterval( anInterval )

        if aStartTime != None:
            self.setStartTime ( aStartTime )

        if anEndTime != None:
            self.setEndTime ( anEndTime )

        if fileType != None:
            self.setDataFileType ( fileType )
        
        self.saveData( None )
                
    def setDirectory ( self, aDirectory ):
        """
        sets directory field to aDirectory
        returns None
        """
        self['directory_entry'].set_text( str( aDirectory ) )

    def getDirectory ( self ):
        """ returns directory choosen by the user """
        return self['directory_entry'].get_text()

    def setInterval ( self, anInterval ):
        """
        sets Interval field of Loggerwindow to anInterval 
        also sets interval checkbox True
        """
        self['datainterval_spinbutton'].set_value( anInterval )
        self['datainterval_checkbox'].set_active( True )
        self.updateDataInterval()

    def getInterval ( self ):
        """
        returns -1 if Interval checkbox is off
        rerurns interval set by user if interva check box is on
        """
        if self['datainterval_checkbox'].get_active( ):
            return self['datainterval_spinbutton'].get_value( )
        else:
            return -1

    def setStartTime ( self, aTime ):
        """
        sets StartTime field of Loggerwindow to anInterval 
        also sets time checkbox True
        """
        self['start_spinbutton'].set_value( aTime )
        self['time_checkbox'].set_active( True )
        self.updateStartEnd()
            
    def getStartTime ( self ):
        """
        returns -1 if Interval checkbox is off
        rerurns interval set by user if interva check box is on
        """
        if self['time_checkbox'].get_active( ):
            return self['start_spinbutton'].get_value( )
        else:
            return -1

    def setEndTime ( self, aTime ):
        """
        sets EndTime field of Loggerwindow to anInterval 
        also sets time checkbox True
        """
        self['end_spinbutton'].set_value( aTime )
        self['time_checkbox'].set_active( True )
        self.updateStartEnd()

    def getEndTime ( self ):
        """
        returns -1 if Interval checkbox is off
        returns interval set by user if interva check box is on
        """
        if self['time_checkbox'].get_active( ):
            return self['end_spinbutton'].get_value( )
        else:
            return -1

    def getDataFileType (self ):
        """
        returns Data Type of file choosen by the user
        """
        return self["datatype_combo"].get_text()

    def setDataFileType (self, aDataType ):
        """
        sets the Datatype of save file to the window
        aDataType can only be 'ecd' or 'binary'
        """
        if aDataType == 'ecd' or aDataType == 'binary':
            self["datatype_combo"].set_text( aDataType )

    def setUseDefaultInterval ( self, aBoolean ):
        """ sets interval checkbox to aBoolean """
        if aBoolean == True or aBoolean == False:
            self['datainterval_checkbox'].set_active( aBoolean )
            self.updateDataInterval()

    def setUseDefaultTime ( self, aBoolean ):
        """ sets time checkbox to aBoolean """
        if aBoolean == True or aBoolean == False:
            self['time_checkbox'].set_active( aBoolean )
            self.updateStartEnd()

    def getUseDefaultInterval ( self ):
        """ return state of interval checkbox  """
        return self['datainterval_checkbox'].get_active( )

    def getUseDefaultTime ( self ):
        """ return state of time checkbox  """
        return self['time_checkbox'].get_active( )

# ---------------------------------------------------------------
# PopupMenu -> gtk.Menu
# ---------------------------------------------------------------
class PopupMenu( gtk.Menu ):
    EDIT_POLICY = 'edit policy'
    def __init__( self, aParent ):
        """
        Constructor
        
        aParent        : parent plugin window
        return -> None
        This method is throwable exception.
        """
        # calls the constructor of super class
        gtk.Menu.__init__(self)

        aMaxStringLength = 0
        aMenuSize = 0
        # sets arguments to instance valiables
        self.theParent = aParent
        # initializes menu item
        self.theMenuItem = {}

        editPolicy = gtk.MenuItem(self.EDIT_POLICY)
        editPolicy.connect('activate', self.theParent.editPolicy )
        editPolicy.set_name(self.EDIT_POLICY)
        self.append( editPolicy )

    def popup(self, pms, pmi, func, button, time):
        gtk.Menu.popup(self, pms, pmi, func, button, time)
        self.show_all()
