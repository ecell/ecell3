#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import os
import gtk
import gobject

import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoPluginWindow import *
from ecell.ui.osogo.VariableReferenceEditor import *
from ecell.ui.osogo.FullPNQueue import *
from ecell.ecssupport import *

# column index of clist
GETABLE_COL   = 0
SETTABLE_COL  = 1
PROPERTY_COL  = 2
VALUE_COL     = 3

PROPERTY_COL_TYPE=gobject.TYPE_STRING
VALUE_COL_TYPE=gobject.TYPE_STRING
GETABLE_COL_TYPE=gobject.TYPE_BOOLEAN
SETTABLE_COL_TYPE=gobject.TYPE_BOOLEAN

PROCESS_DISCARD_LIST=[ 'Name' ]
VARIABLE_DISCARD_LIST=[ 'Name', 'MolarConc', 'NumberConc' ]
SYSTEM_DISCARD_LIST=[ 'Name' ]



class PropertyWindow(OsogoPluginWindow):

    # ---------------------------------------------------------------
    # constructor
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __init__( self, aDirName, aData, aPluginManager, rootWidget=None ):
        
        # calls superclass's constructor
        OsogoPluginWindow.__init__( self, aDirName, aData,
                                   aPluginManager, rootWidget=rootWidget )
        self.theStatusBarWidget = None
        self.theQueue = None
        self.theAssociatedSession = None

    # end of __init__

    def setParent ( self, aParent ):
        self.theParent = aParent


    def openWindow( self ):
        #self.openWindow()
        OsogoPluginWindow.openWindow(self)
        
        # add handers
        self.addHandlers( { 'on_checkViewAll_toggled' : self.updateViewAllProperties } )

        # initializes buffer
        self.thePreFullID = None
        self.thePrePropertyMap = {}
       
        # initializes ListStore
        self.theListStore=gtk.ListStore(
                                        GETABLE_COL_TYPE,
                                        SETTABLE_COL_TYPE,
                                        PROPERTY_COL_TYPE,
                                        VALUE_COL_TYPE )
        self.lockCursor = False
        self.cursorHandler = self['theTreeView'].connect('cursor_changed', self.__cursorChanged)
        self['theTreeView'].connect('button_press_event', self.__popupMenu)
        
        self['theTreeView'].set_model(self.theListStore)
        
        renderer=gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( "R", renderer, active = GETABLE_COL )
        column.set_visible( True )
        column.set_resizable( True )
        column.set_sort_column_id( GETABLE_COL )
        column.set_reorderable( True )
        self['theTreeView'].append_column(column)

        renderer=gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( "W", renderer, active = SETTABLE_COL )
        column.set_visible( True )
        column.set_reorderable( True )
        column.set_sort_column_id( SETTABLE_COL )
        column.set_resizable( True )
        self['theTreeView'].append_column(column)

        renderer=gtk.CellRendererText()
        column=gtk.TreeViewColumn( "Property", renderer, text=PROPERTY_COL)
        column.set_visible( True )
        column.set_resizable( True )
        column.set_reorderable( True )
        column.set_sort_column_id( PROPERTY_COL )
        self['theTreeView'].append_column(column)

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__valueEdited)
        column=gtk.TreeViewColumn( "Value", renderer, text=VALUE_COL,
                                  editable=SETTABLE_COL )
        column.set_visible( True )
        column.set_sizing( 1 ) # auto sizing
        self['theTreeView'].append_column(column)
        column.set_sort_column_id( VALUE_COL )
        column.set_reorderable( True )
        self.theValueColumn = column

        # creates popu menu
        self.thePopupMenu = PropertyWindowPopupMenu( self.thePluginManager, self )
        # initializes statusbar
        self.theStatusBarWidget = self['statusbar']
        self.theVarrefTabNumber  = -1
        self.theNoteBook = self['notebookProperty']
        self.theVarrefEditor = None

        self['entrySystemSubsystems'].set_property( 'xalign', 1 )
        self['entrySystemProcesses'].set_property( 'xalign', 1 )
        self['entrySystemVariables'].set_property( 'xalign', 1 )

        self['entryProcessVarRefTotal'].set_property( 'xalign', 1 )
        self['entryProcessVarRefPositive'].set_property( 'xalign', 1 )
        self['entryProcessVarRefZero'].set_property( 'xalign', 1 )
        self['entryProcessVarRefNegative'].set_property( 'xalign', 1 )

        self['entryVariableValue'].set_property( 'xalign', 1 )
        self['entryVariableVelocity'].set_property( 'xalign', 1 )
        self['entryVariableMolar'].set_property( 'xalign', 1 )
        self['entryVariableNumber'].set_property( 'xalign', 1 )

        if len( self.theRawFullPNList ) == 0:
            return
        # set default as not to view all properties
        self['checkViewAll'].set_active( False )

        self.setIconList(
            os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )
        #self.__setFullPNList()
        self.update()

        if ( len( self.getFullPNList() ) > 1 ) and ( rootWidget !=
                                                    'EntityWindow' ):
            self.thePreFullID, _ = convertFullPNToFullID( self.getFullPN() )
            aClassName = self.__class__.__name__

        # registers myself to PluginManager
        self.thePluginManager.appendInstance( self ) 
        



    # =====================================================================
    def setStatusBar( self, aStatusBarWidget ):
        """sets a status bar to this window. 
        This method is used when this window is displayed on other window.
        aStatusBarWidget  --  a status bar (gtk.StatusBar)
        Returns None
        [Note]:The type of aStatusBarWidget is wrong, throws exception.
        """

        if type(aStatusBarWidget) != gtk.Statusbar:
            raise TypeError("%s must be gtk.StatusBar.")

        self.theStatusBarWidget = aStatusBarWidget


    # =====================================================================
    def clearStatusBar( self ):
        """clear messaeg of statusbar
        """

        self.theStatusBarWidget.push(1,'')


    # =====================================================================
    def showMessageOnStatusBar( self, aMessage ):
        """show messaegs on statusbar
        aMessage   --  a message to be displayed on statusbar (str)
        [Note]:message on statusbar should be 1 line. If the line aMessage is
               more than 2 lines, connects them as one line.
        """

        aMessage = ', '.join( aMessage.split( '\n' ) )

        self.theStatusBarWidget.push(1,aMessage)


    # ---------------------------------------------------------------
    # Overwrite Window.__getitem__
    # When this instance is on EntityListWindow,
    # self['statusbar'] returns the statusbar of EntityListWindow
    #
    # aKey  : a key to access widget
    # return -> a widget
    # ---------------------------------------------------------------
    def __getitem__( self, aKey ):

        # When key is not statusbar, return default widget
        if aKey != 'statusbar':
                return self.widgets.get_widget( aKey )

        # When is 'statusbar' and self.setStatusBar method has already called,
        # returns, the statusbar of EntityWindow.
        else:
            if self.theStatusBarWidget != None:
                return self.theStatusBarWidget
            else:
                return None

    # end of __getitem__

    # ---------------------------------------------------------------
    # Overwrite Window.setRawFullPNList
    # This method is used by EntityListWindow
    # change RawFullPNList
    #
    # aRawFullPNList  : a RawFullPNList
    # return -> None
    # ---------------------------------------------------------------
    def setRawFullPNList( self, aRawFullPNList ):
        # When aRawFullPNList is not changed, does nothing.
        if self.theRawFullPNList == aRawFullPNList:
            # do nothing
            pass

        # When aRawFullPNList is changed, updates its and call self.update().
        else:
            self['theTreeView'].disconnect( self.cursorHandler )
            OsogoPluginWindow.setRawFullPNList( self, aRawFullPNList )
            self.update()
            self.cursorHandler = self['theTreeView'].connect('cursor_changed', self.__cursorChanged)



       
    # end of setRawFullPNList

    # ---------------------------------------------------------------
    # update (overwrite the method of superclass)
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def update( self ):
        aFullID = None
        aFullPN = None
        fullUpdate = False

        if self.theSession.theSession is not self.theAssociatedSession:
            self.theRawFullPNList = []
            if self.theSession.theSession is not None:
                if self.theParent != None:
                    if self.theParent.__class__.__name__ == "EntityListWindow":
                        self.theQueue = self.theParent.getQueue()
                    else:
                        self['navigation'].visible = True
                        self.theQueue = FullPNQueue( ( self['backbutton'], self['forwardbutton'] ), self.theRawFullPNList )
                else:
                    self['navigation'].visible = True
                    self.theQueue = FullPNQueue( ( self['backbutton'], self['forwardbutton'] ), self.theRawFullPNList )
                self.theAssociatedSession = self.theSession.theSession
                self.theQueue.registerCallback( self.setRawFullPNList )
            else:
                self.theQueue = None
        else:
            aFullPN = self.getFullPN()
            if aFullPN is not None:
                aFullID, _ = convertFullPNToFullID( self.getFullPN() )
        if self.thePreFullID != aFullID:
            fullUpdate = True

        # ----------------------------------------------------
        # updates widgets
        # ----------------------------------------------------
        # creates EntityStub
        if aFullID is not None:
            anEntityStub = self.theSession.createEntityStub( createFullIDString( aFullID ) )
        else:
            anEntityStub = None
            fullUpdate = True

        if not fullUpdate:
            # gets propery values for thePreProperyMap in case value is not tuple
            for aPropertyName in self.thePrePropertyMap.keys():
                aProperty = self.thePrePropertyMap[aPropertyName]
                if type( aProperty[0] ) not in ( tuple, list ):
                    aProperty[0] = anEntityStub.getProperty( aPropertyName )
            if self.theVarrefEditor != None:
                self.theVarrefEditor.update()
        else:
            self.theSelectedFullPN = ''

            # -----------------------------------------------
            # updates each widget
            # Type, ID, Path, Classname
            # -----------------------------------------------
            if anEntityStub is not None:
                anEntityType = ENTITYTYPE_STRING_LIST[aFullPN[TYPE]]
                anID = aFullPN[ID]
                aSystemPath = str( aFullPN[SYSTEMPATH] )

                self['entryClassName'].set_text( anEntityStub.getClassname() )
                self['entryFullID'].set_text( ':'.join( [ anEntityType,
                                                          aSystemPath,
                                                          anID ] ) )
                # saves properties to buffer
                self.thePrePropertyMap = {}
                for aProperty in anEntityStub.getPropertyList():
                    self.thePrePropertyMap[str(aProperty)] = [None, None]
                    self.thePrePropertyMap[str(aProperty)][0] =\
                            anEntityStub.getProperty(aProperty)
                    self.thePrePropertyMap[str(aProperty)][1] =\
                            anEntityStub.getPropertyAttributes(aProperty)

                self['entryName'].set_text( str(
                                     self.thePrePropertyMap[ 'Name' ][0] )  )

                # save current full id to previous full id.
                self.thePreFullID = aFullID
                self.setSelectedFullPN( aFullPN )
            else:
                self['entryClassName'].set_text( "" )
                self['entryClassName'].set_max_length(0)
                self['entryFullID'].set_text( "" )
                self['entryName'].set_text( "" )

            # update Summary tab for unique fields of each entity type
            # update the respective Entity's PropertyList
            if aFullPN is not None:
                self.__setDiscardList()
                if aFullPN[TYPE] == PROCESS:
                    self.__createVariableReferenceListTab()
                    self.__updateProcess()
                elif aFullPN[TYPE] == VARIABLE:
                    self.__deleteVariableReferenceListTab()
                    self.__updateVariable()
                elif aFullPN[TYPE] == SYSTEM:
                    self.__deleteVariableReferenceListTab()
                    self.__updateSystem()

        # updates status bar
        if self['statusbar'] != None:
            self['statusbar'].push(1,'')

                
    def __updatePropertyList( self ):
        self.theList = []
        aPropertyList = self.thePrePropertyMap.keys()

        # do nothing for following properties
        try:
            aPropertyList.remove( 'FullID' )
            aPropertyList.remove( 'Name' )
        except:
            pass

        for aPropertyName in aPropertyList: # for (1)
            if aPropertyName not in self.theDiscardList:
                aProperty = self.thePrePropertyMap[aPropertyName]
                anAttribute = aProperty[1]

                # When the getable attribute is false, value is ''
                if anAttribute[GETABLE] == False:
                    aValue = ''
                else:
                    aValue = str( aProperty[0] )


                self.theList.append( [
                                      anAttribute[GETABLE],
                                      anAttribute[SETTABLE],
                                      aPropertyName,
                                      aValue ] )
        lockCursor = self.lockCursor
        self.lockCursor = True
        self['theTreeView'].get_selection().unselect_all()

#        self.theListStore.clear()
        anIter = self.theListStore.get_iter_first()
        #first rewrite properties

        for aValue in self.theList:
            if anIter == None:
                anIter=self.theListStore.append( )
            cntr = 0
            for valueitem in aValue:
                self.theListStore.set_value(anIter,cntr,valueitem)
                cntr += 1
            anIter = self.theListStore.iter_next( anIter )
        while anIter != None:
            nextIter = self.theListStore.iter_next(anIter)
            self.theListStore.remove( anIter )
            anIter = nextIter

        if len( self.theRawFullPNList ) > 0:
            self.setSelectedFullPN( self.theRawFullPNList[0] )

        self.lockCursor = lockCursor


    def __updateProcess( self ):
        self.__updatePropertyList() 
        aVariableReferenceList = self.thePrePropertyMap[
                                       'VariableReferenceList'][0]
        aPositiveCoeff = 0
        aZeroCoeff = 0
        aNegativeCoeff = 0
        for aVariableReference in aVariableReferenceList:
            if aVariableReference[2] == 0:
                aZeroCoeff = aZeroCoeff + 1
            elif aVariableReference[2] > 0:
                aPositiveCoeff = aPositiveCoeff + 1
            elif aVariableReference[2] < 0:
                aNegativeCoeff = aNegativeCoeff + 1

        aStepperID = str( self.thePrePropertyMap[ 'StepperID' ][0] )
        anActivity = str( self.thePrePropertyMap[ 'Activity' ][0] )
        isContinuous = bool( self.thePrePropertyMap[ 'IsContinuous'][0] )
        aPriority = str( self.thePrePropertyMap[ 'Priority'][0] )

        self['entryProcessVarRefTotal'].set_text(
                                         str( len( aVariableReferenceList ) ) )
        self['entryProcessVarRefPositive'].set_text( str( aPositiveCoeff ) )
        self['entryProcessVarRefZero'].set_text( str( aZeroCoeff ) )
        self['entryProcessVarRefNegative'].set_text( str( aNegativeCoeff ) )
        self['entryProcessStepper'].set_text( aStepperID )
        self['entryProcessActivity'].set_text( anActivity )
        self['entryProcessIsContinuous'].set_text( str( isContinuous ) )
        self['entryProcessPriority'].set_text( aPriority )

        self['systemFrame'].hide()
        self['processFrame'].show()
        self['variableFrame'].hide()


        

    def __updateVariable( self ):
        self.__updatePropertyList()
        aMolarConc = str( self.thePrePropertyMap[ 'MolarConc' ][0] )
        aValue = str( self.thePrePropertyMap[ 'Value' ][0] )
        aNumberConc = str( self.thePrePropertyMap[ 'NumberConc' ][0] )
        aVelocity = str( self.thePrePropertyMap[ 'Velocity' ][0] )
        if self.thePrePropertyMap[ 'Fixed'][0]  == 1:
            aFixed = '<b>Fixed</b>'
        else:
            aFixed = ''
        
        self['entryVariableValue'].set_text( aValue )
        self['entryVariableMolar'].set_text( aMolarConc )
        self['entryVariableNumber'].set_text( aNumberConc )
        self['labelVariableQuantities'].set_markup( 'Quantities / Concentration %s' % aFixed )
        self['entryVariableVelocity'].set_text( aVelocity )

        self['systemFrame'].hide()
        self['processFrame'].hide()
        self['variableFrame'].show()

    def __updateSystem( self ):
        self.__updatePropertyList()
        aFullID, _ = convertFullPNToFullID( self.getFullPN() )
        aSystemPath = createSystemPathFromFullID( aFullID )
        aProcessList = self.theSession.getEntityList( 'Process', aSystemPath )
        aVariableList = self.theSession.getEntityList( 'Variable', aSystemPath )
        aSystemList = self.theSession.getEntityList( 'System', aSystemPath ) 
        aStepperID = str( self.thePrePropertyMap[ 'StepperID' ][0] )
        aSize = str( self.thePrePropertyMap[ 'Size' ][0] )

        self['entrySystemSubsystems'].set_text( str( len( aSystemList ) ) )
        self['entrySystemProcesses'].set_text( str( len( aProcessList ) ) )
        self['entrySystemVariables'].set_text( str( len( aVariableList ) ) )
        self['entrySystemStepper'].set_text( aStepperID )
        self['entrySystemSize'].set_text( aSize )
       
        self['systemFrame'].show()
        self['processFrame'].hide()
        self['variableFrame'].hide()

    def updateViewAllProperties( self, *anObject ):
        self.__setDiscardList()
        self.__updatePropertyList()


    def __setDiscardList( self ):
        aFullPN = self.getFullPN()
        if aFullPN is None:
            return
        isViewAll = self['checkViewAll'].get_active()
        if isViewAll:
            self.theDiscardList = []
        else:
            if aFullPN[TYPE] == PROCESS:
                self.theDiscardList = PROCESS_DISCARD_LIST 
            elif aFullPN[TYPE] == VARIABLE:
                self.theDiscardList = VARIABLE_DISCARD_LIST 
            elif aFullPN[TYPE] == SYSTEM:
                self.theDiscardList = SYSTEM_DISCARD_LIST 
        

    def __valueEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """
        
        aNewValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        aSelectedProperty = self.theListStore.get_value( anIter, PROPERTY_COL )
        aFullID, _ = convertFullPNToFullID( self.getFullPN() )
        self.theSelectedFullPN = convertFullIDToFullPN( aFullID, aSelectedProperty )
        
        # disable VariableReferenceList editing because of a bug when
        # saving changes
        if aSelectedProperty != 'VariableReferenceList':
            self.__updateValue( aNewValue, anIter, VALUE_COL )

    

    # ---------------------------------------------------------------
    # updateValue
    #   - sets inputted value to the simulator
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __updateValue( self, aValue, anIter, aColumn ):


        # ------------------------------------
        # gets getable status
        # ------------------------------------
        aGetable = self.theListStore.get_value( anIter, GETABLE_COL )

        # ------------------------------------
        # checks the type of inputted value 
        # ------------------------------------
        if aGetable:
            aPreValue = self.theListStore.get_value( anIter, aColumn )

            # ------------------------------------
            # when type is integer
            # ------------------------------------
            if type(aPreValue) == type(0):
                try:
                    aValue = int(aValue)
                except:
                    import sys
                    import traceback
                    anErrorMessage = '\n'.join(
                        traceback.format_exception( 
                            sys.exc_type,sys.exc_value,sys.exc_traceback ) )
                    self.theSession.message("-----An error happens.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input an integer!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None

            # ------------------------------------
            # when type is float
            # ------------------------------------
            elif type(aPreValue) == type(0.0):
                try:
                    aValue = float(aValue)
                except:
                    import sys
                    import traceback
                    anErrorMessage = '\n'.join( traceback.format_exception(  sys.exc_type,sys.exc_value,sys.exc_traceback ) )
                    self.theSession.message("-----An error happened.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input a float!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None

            # ------------------------------------
            # when type is tuple
            # ------------------------------------
            elif type(aPreValue) == type(()):
                try:
                    aValue = convertStringToTuple( aValue )

                except:
                    import sys
                    import traceback
                    anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
                    self.theSession.message("-----An error happens.-----")
                    self.theSession.message(anErrorMessage)
                    self.theSession.message("---------------------------")

                    # creates and display error message dialog.
                    anErrorMessage = "Input a tuple!"
                    anErrorTitle = "The type error!"
                    if self['statusbar'] != None:
                        self['statusbar'].push(1,anErrorMessage)
                    anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
                    return None


        try:
            self.theSession.setEntityProperty( self.theSelectedFullPN, aValue ) 
            self.theSession.updateWindows()
            self.lockCursor = True
            self['theTreeView'].get_selection().select_iter( anIter )
            self.lockCursor = False
        except:


            import sys
            import traceback
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
            self.theSession.message("-----An error happens.-----")
            self.theSession.message(anErrorMessage)
            self.theSession.message("---------------------------")

            # creates and display error message dialog.
            anErrorMessage = "An error happened! See MessageWindow."
            anErrorTitle = "An error happened!"
            if self['statusbar'] != None:
                self['statusbar'].push(1,anErrorMessage)
            anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
        else:

            self.__updatePropertyList()
            #self.thePluginManager.updateAllPluginWindow() 

    # end of updateValue


    def getSelectedFullPN( self ):

        anIter = self['theTreeView'].get_selection().get_selected()[1]
        if anIter == None:
			self.theSelectedFullPN = ''
        else:
            aSelectedProperty = self.theListStore.get_value( anIter,
                                                            PROPERTY_COL )
            aFullID, _ = convertFullPNToFullID( self.getFullPN() )
            self.theSelectedFullPN = convertFullIDToFullPN( aFullID, aSelectedProperty )
        return self.theSelectedFullPN

    def setSelectedFullPN( self, aFullPN ):

        aPropertyName = aFullPN[3]
        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            if self.theListStore.get_value( anIter, PROPERTY_COL ) == aPropertyName:
                lockCursor = self.lockCursor
                self.lockCursor = True
                self['theTreeView'].get_selection().select_iter( anIter )
                self.lockCursor = lockCursor
                break
            anIter = self.theListStore.iter_next( anIter ) 
                
    def __cursorChanged( self, *args ):

        if self.lockCursor:
            return

        aFullPNList = [ self.getSelectedFullPN() ]
        self.lockCursor = True
        self.theQueue.pushFullPNList( aFullPNList )
        self.lockCursor = False

    # ---------------------------------------------------------------
    # popupMenu
    #   - show popup menu
    #
    # aWidget         : widget
    # anEvent          : an event
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def __popupMenu( self, aWidget, anEvent ):

        if anEvent.button == 3:  # 3 means right

            if self['theTreeView'].get_selection().get_selected()[1]==None :
                return None

            self.thePopupMenu.popup( None, None, None, 1, 0 )

    # end of poppuMenu

    def createNewPluginWindow( self, anObject ):

        # gets PluginWindowName from selected MenuItem
        aPluginWindowName = anObject.get_name()

        # creates PluginWindow
        self.thePluginManager.createInstance( aPluginWindowName, self.theRawFullPNList )

    # end of createNewPluginWindow

    def __createVariableReferenceListTab( self ):
        aFullID, _ = convertFullPNToFullID( self.getFullPN() )
        if self.theVarrefTabNumber  != -1:
            if self.theVarrefEditor.getProcessFullID() == aFullID:
                return
            else:
                self.theVarrefEditor.setDisplayedFullID( aFullID )
        else:
            aFrame = gtk.Frame()
            aFrame.show()
            aLabel = gtk.Label("Variable References")
            aLabel.show()
            self.theNoteBook.append_page(  aFrame, aLabel )
            self.theVarrefTabNumber = 2
            self.theVarrefEditor = VariableReferenceEditor( self, aFrame )


    def __deleteVariableReferenceListTab( self ):
        if self.theVarrefTabNumber  == -1:
            return
        curPage = self.theNoteBook.get_current_page()
        if  curPage == self.theVarrefTabNumber:
            curPage = 0

        self.theNoteBook.remove_page( self.theVarrefTabNumber )
        self.theNoteBook.set_current_page( curPage )
        self.theVarrefTabNumber  = -1
        self.theVarrefEditor = None


# ----------------------------------------------------------
# PropertyWindowPopupMenu -> gtk.Menu
#   - popup menu used by property window
# ----------------------------------------------------------
class PropertyWindowPopupMenu( gtk.Menu ):

    # ----------------------------------------------------------
    # Constructor
    #   - added PluginManager reference
    #   - added OsogoPluginWindow reference
    #   - acreates all menus
    #
    # aPluginManager : reference to PluginManager
    # aParent        : property window
    #
    # return -> None
    # This method is throwabe exception.
    # ----------------------------------------------------------
    def __init__( self, aPluginManager, aParent ):

        gtk.Menu.__init__(self)

        self.theParent = aParent
        self.thePluginManager = aPluginManager
        self.theMenuItem = {}

        # ------------------------------------------
        # initializes the size of menu
        # ------------------------------------------
        aMaxStringLength = 0
        aMenuSize = 0

        # ------------------------------------------
        # adds plugin window
        # ------------------------------------------
        for aPluginMap in self.thePluginManager.thePluginMap.keys():
            self.theMenuItem[aPluginMap]= gtk.MenuItem(aPluginMap)
            self.theMenuItem[aPluginMap].connect('activate', self.theParent.createNewPluginWindow )
            self.theMenuItem[aPluginMap].set_name(aPluginMap)
            self.append( self.theMenuItem[aPluginMap] )
            if aMaxStringLength < len(aPluginMap):
                aMaxStringLength = len(aPluginMap)
            aMenuSize += 1

        self.theWidth = (aMaxStringLength+1)*8
        #self.theHeight = (aMenuSize+1)*21 + 3
        self.theHeight = (aMenuSize+1)*21 + 3
        #self.set_usize( self.theWidth, self.theHeight )

        self.set_size_request( self.theWidth, self.theHeight )
        #self.append( gtk.MenuItem() )
        #self.set_size_request( 150, 450 )

    # end of __init__


    # ---------------------------------------------------------------
    # popup
    #    - shows this popup memu
    #
    # return -> None
    # This method is throwable exception.
    # ---------------------------------------------------------------
    def popup(self, pms, pmi, func, button, time):

        # shows this popup memu
        gtk.Menu.popup(self, pms, pmi, func, button, time)
        self.show_all()

    # end of poup


# end of OsogoPluginWindowPopupMenu







if __name__ == "__main__":


    class simulator:

        dic={'PropertyList':
             ('PropertyList', 'ClassName', 'A','B','C','Substrate','Product'),
             'ClassName': ('MichaelisMentenProcess', ),
             'A': ('aaa', ) ,
             'B': (1.04E-3, ) ,
             'C': (41, ),
             'Substrate': ('Variable:/CELL/CYTOPLASM:ATP',
                           'Variable:/CELL/CYTOPLASM:ADP',
                           'Variable:/CELL/CYTOPLASM:AMP',
                           ),
             'Product': ('Variable:/CELL/CYTOPLASM:GLU',
                         'Variable:/CELL/CYTOPLASM:LAC',
                         'Variable:/CELL/CYTOPLASM:PYR',
                         ),
             'PropertyAttributes' : ('1','2','3','4','5','6','7','8'),
             } 

        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn[PROPERTY]]
    
    fpn = FullPropertyName('Process:/CELL/CYTOPLASM:MichaMen:PropertyName')



    def mainQuit( obj, data ):
        gtk.main_quit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.main()

    def main():
        aPluginManager = Plugin.PluginManager()
        aPropertyWindow = PropertyWindow( 'plugins', simulator(), [fpn,] ,aPluginManager)
        aPropertyWindow.addHandler( 'gtk_main_quit', mainQuit )
        aPropertyWindow.update()

        mainLoop()

    main()




