#!/usr/bin/env python
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

import os
import string
import gobject
import gtk

from ecell.ecs_constants import *
from ecell.ui.osogo.constants import *
from ecell.ui.osogo.OsogoPluginWindow import OsogoPluginWindow
from ecell.ui.osogo.EntityListWindow import EntityListWindow
from ecell.ui.osogo.VariableReferenceEditor import *
from ecell.ui.osogo.FullPNQueue import *
import ecell.ui.osogo.config as config
import ecell.util as util
from ecell.ui.osogo.utils import *

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
    def __init__( self, aDirName, aData, aPluginManager ):
        """"
        constructor
        
        return -> None
        This method can throw an exception.
        """
        # calls superclass's constructor
        OsogoPluginWindow.__init__(
            self, aDirName, aData, aPluginManager.theSession )
        self.theStatusBarWidget = None
        self.theQueue = None

    def _setParentIntn( self, aParent, aParentWidgetName ):
        if aParent != None and aParent.__class__ == EntityListWindow:
            if self.theQueue != None:
                self.theQueue.unregisterCallback( self.setRawFullPNList )
            self.theQueue = aParent.getQueue()
            self.theQueue.registerCallback( self.setRawFullPNList )
        else:
            if self.theParent != None or self.theQueue == None:
                if self.theQueue != None:
                    self.theQueue.unregisterCallback( self.setRawFullPNList )
                aQueue = FullPNQueue( self.theRawFullPNList )
                self.addChild( aQueue, 'navigator_area' )
                aQueue.initUI()
                aQueue.registerCallback( self.setRawFullPNList )
                self.theQueue = aQueue
        OsogoPluginWindow._setParentIntn( self, aParent, aParentWidgetName )

    def initUI( self ):
        OsogoPluginWindow.initUI( self )

        # ugly... but works
        if self.theParent == None:
            self._setParentIntn( None, None )

        # add handers
        self.addHandlers(
            {
                'on_checkViewAll_toggled' : self.updateViewAllProperties
                }
            )

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
        column = gtk.TreeViewColumn( "Value", renderer, text=VALUE_COL,
                                  editable=SETTABLE_COL )
        column.set_visible( True )
        column.set_sizing( 1 ) # auto sizing
        self['theTreeView'].append_column(column)
        column.set_sort_column_id( VALUE_COL )
        column.set_reorderable( True )
        self.theValueColumn = column

        # creates popu menu
        self.thePopupMenu = PropertyWindowPopupMenu( self )
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

        if self.theRawFullPNList == ():
            return
        # set default as not to view all properties
        self['checkViewAll'].set_active( False )

        self.setIconList(
            os.path.join( config.glade_dir, "ecell.png" ),
            os.path.join( config.glade_dir, "ecell32.png" ) )

        if len( self.getFullPNList() ) > 1 and rootWidget != 'EntityWindow':
            self.thePreFullID = self.getFullID()
            aClassName = self.__class__.__name__
        self.setTitle( "%s - %s" % (
            self.getName(),
            str( self.getFullPN().fullID ) ) )
        self.update()

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

    def clearStatusBar( self ):
        """clear messaeg of statusbar
        """

        self.theStatusBarWidget.push(1,'')

    def showMessageOnStatusBar( self, aMessage ):
        """
        show messaegs on statusbar
        aMessage   --  a message to be displayed on statusbar (str)
        [Note]:message on statusbar should be 1 line. If the line aMessage is
               more than 2 lines, connects them as one line.
        """

        aMessage = string.join( string.split(aMessage,'\n'), ', ' )

        self.theStatusBarWidget.push(1,aMessage)

    def setRawFullPNList( self, aRawFullPNList ):
        """
        Overwrite Window.setRawFullPNList
        This method is used by EntityListWindow
        change RawFullPNList
        
        aRawFullPNList  : a RawFullPNList
        return -> None
        """
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

    def update( self, fullUpdate = False ):
        """
        update (overwrite the method of superclass)
        
        return -> None
        This method can throw an exception.
        """

        if self.theSession.theModelWalker == None:
            return

        aFullID = self.getFullPN().fullID
        if aFullID == None:
            return

        # checks a value is changed or not
        if self.thePreFullID != aFullID:
            fullUpdate = True

        # creates EntityStub
        anEntityStub = self.theSession.createEntityStub( aFullID )

        if fullUpdate == False:
            # gets propery values for thePreProperyMap in case value is not tuple
            for aPropertyName in self.thePrePropertyMap.keys():
                aProperty = self.thePrePropertyMap[aPropertyName]
                if type( aProperty[0] ) not in ( type( () ), type( [] ) ):
                    aProperty[0] = anEntityStub.getProperty(aPropertyName)
            if self.theVarrefEditor != None:
                self.theVarrefEditor.update()
                
        else:

            self.theSelectedFullPN = ''

            # -----------------------------------------------
            # updates each widget
            # Type, ID, Path, Classname
            # -----------------------------------------------
            anEntityType = self.getFullPN().fullID.getTypeName()

            self['labelEntityType'].set_text( anEntityType + ' Property' )
            self['entryClassName'].set_text( anEntityStub.getClassname() )
            self['entryFullID'].set_text( str( self.getFullPN().fullID ) )

            # saves properties to buffer
            self.thePrePropertyMap = {}
            for aProperty in anEntityStub.getPropertyList():
                self.thePrePropertyMap[str(aProperty)] = [None, None]
                self.thePrePropertyMap[str(aProperty)][0] =\
                        anEntityStub.getProperty(aProperty)
                self.thePrePropertyMap[str(aProperty)][1] =\
                        anEntityStub.getPropertyAttributes(aProperty)
                
            # update Summary tab for unique fields of each entity type
            # update the respective Entity's PropertyList
            self.__setDiscardList()
            if self.getFullPN().fullID.typeCode == PROCESS:
                self.showVariableReferenceListTab()
                self.__updateProcess()
            elif self.getFullPN().fullID.typeCode == VARIABLE:
                self.hideVariableReferenceListTab()
                self.__updateVariable()
            elif self.getFullPN().fullID.typeCode == SYSTEM:
                self.hideVariableReferenceListTab()
                self.__updateSystem()


            self['entryName'].set_text( str(
                                 self.thePrePropertyMap[ 'Name' ][0] )  )

        # save current full id to previous full id.
        self.preFullID = self.getFullPN().fullID
        self.setSelectedFullPN(self.theRawFullPNList[0])
        # updates status bar
        if self.theStatusBarWidget != None:
            self.theStatusBarWidget.push(1,'')
                
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
            aFixed = '(Fixed)'
        else:
            aFixed = '(Not Fixed)'
        
        self['entryVariableValue'].set_text( aValue )
        self['entryVariableMolar'].set_text( aMolarConc )
        self['entryVariableNumber'].set_text( aNumberConc )
        self['labelVariableQuantities'].set_markup( '<b>Quantities '
                                                   + aFixed + '</b>' )
        self['entryVariableVelocity'].set_text( aVelocity )

        self['systemFrame'].hide()
        self['processFrame'].hide()
        self['variableFrame'].show()

    def __updateSystem( self ):
        self.__updatePropertyList()
        aSystemPath = self.getFullPN().fullID.toSystemPath()
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
        isViewAll = self['checkViewAll'].get_active()
        if isViewAll:
            self.theDiscardList = []
        else:
            if self.getFullPN().fullID.typeCode == PROCESS:
                self.theDiscardList = PROCESS_DISCARD_LIST 
            elif self.getFullPN().fullID.typeCode == VARIABLE:
                self.theDiscardList = VARIABLE_DISCARD_LIST 
            elif self.getFullPN().fullID.typeCode == SYSTEM:
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
        self.theSelectedFullPN = self.getFullPN().fullID.createFullPN(
            aSelectedProperty )
        
        # disable VariableReferenceList editing because of a bug when
        # saving changes
        if aSelectedProperty != 'VariableReferenceList':
            self.__updateValue( aNewValue, anIter, VALUE_COL )

    def __updateValue( self, aValue, anIter, aColumn ):
        """
          - sets inputted value to the simulator
        
        return -> None
        This method can throw an exception.
        """
        # gets getable status
        aGetable = self.theListStore.get_value( anIter, GETABLE_COL )

        # checks the type of inputted value 
        if aGetable == True:
            aPreValue = self.theListStore.get_value( anIter, aColumn )
            try:
                # when type is integer
                if type(aPreValue) == type(0):
                    aValue = string.atoi(aValue)
                # when type is float
                elif type(aPreValue) == type(0.0):
                    aValue = string.atof(aValue)
                # when type is tuple
                elif type(aPreValue) == type(()):
                    aValue = convertStringToTuple( aValue )
            except:
                import sys
                # creates and display error message dialog.
                anErrorMessage = "Invalid value format"
                if self.theStatusBarWidget != None:
                    self.theStatusBarWidget.push(1,anErrorMessage)
                showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
                return None
        aFullPNString = str( self.theSelectedFullPN )

        try:
            self.setValue( self.theSelectedFullPN, aValue )
            lockCursor = self.lockCursor
            self.lockCursor = True
            self['theTreeView'].get_selection().select_iter( anIter )
            self.lockCursor = False
        except:


            import sys
            import traceback
            anErrorMessage = string.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
            self.theSession.message("-----An error happens.-----")
            self.theSession.message(anErrorMessage)
            self.theSession.message("---------------------------")

            # creates and display error message dialog.
            anErrorMessage = "An error occurred. See MessageWindow."
            if self.theStatusBarWidget != None:
                self.theStatusBarWidget.push( 1, anErrorMessage )
            showPopupMessage( OK_MODE, anErrorMessage, 'Error' )
        else:

            self.__updatePropertyList()

    def setValue( self, aFullPN, aValue ):

        self.thePrePropertyMap[ aFullPN[ 3 ] ][ 0 ] = aValue
        return OsogoPluginWindow.setValue( self, aFullPN, aValue )

    # end of setValue

    def getSelectedFullPN( self ):

        anIter = self['theTreeView'].get_selection().get_selected()[1]
        if anIter == None:
            self.theSelectedFullPN = ''
        else:
            aSelectedProperty = self.theListStore.get_value( anIter,
                                                            PROPERTY_COL )
            self.theSelectedFullPN = self.getFullPN().fullID.createFullPN(
                aSelectedProperty )
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

    def __popupMenu( self, aWidget, anEvent ):
        """
        popupMenu
          - show popup menu
        
        aWidget         : widget
        anEvent          : an event
        return -> None
        This method can throw an exception.
        """
        if anEvent.button == 3:  # 3 means right

            if self['theTreeView'].get_selection().get_selected()[1]==None :
                return None

            self.thePopupMenu.popup( None, None, None, 1, 0 )

    def createNewPluginWindow( self, anObject ):
        aPluginWindowName = anObject.get_name()
        self.theSession.openPluginWindow( aPluginWindowName, self.theRawFullPNList )

    def showVariableReferenceListTab( self ):
        if self.theVarrefEditor == None:
            aVarrefEditor = VariableReferenceEditor()
            self.addChild( aVarrefEditor, 'varref_area' )
            aVarrefEditor.initUI()
            aVarrefEditor.show()
            self.theVarrefEditor = aVarrefEditor
        if self.theVarrefEditor.getProcessFullID() != self.getFullPN().fullID:
            self.theVarrefEditor.setDisplayedFullID( self.getFullPN().fullID )
        self['varref_area'].show()

    def hideVariableReferenceListTab( self ):
        self['varref_area'].hide()

class PropertyWindowPopupMenu( gtk.Menu ):
    """
    - popup menu used by property window
    """
    def __init__( self, aParent ):
        """
        Constructor
          - added PluginManager reference
          - added OsogoPluginWindow reference
          - acreates all menus
        
        aPluginManager : reference to PluginManager
        aParent        : property window
        
        return -> None
        This method is throwabe exception.
        """
        gtk.Menu.__init__(self)

        self.theParent = aParent
        self.theMenuItem = {}

        # initializes the size of menu
        aMaxStringLength = 0
        aMenuSize = 0

        # adds plugin window
        for aPluginName in self.theParent.theSession.getLoadedModules():
            self.theMenuItem[aPluginName]= gtk.MenuItem( aPluginName )
            self.theMenuItem[aPluginName].connect('activate',
                    self.theParent.createNewPluginWindow )
            self.theMenuItem[aPluginName].set_name(aPluginName)
            self.append( self.theMenuItem[aPluginName] )
            if aMaxStringLength < len(aPluginName):
                aMaxStringLength = len(aPluginName)
            aMenuSize += 1

        self.theWidth = (aMaxStringLength+1)*8
        #self.theHeight = (aMenuSize+1)*21 + 3
        self.theHeight = (aMenuSize+1)*21 + 3
        #self.set_usize( self.theWidth, self.theHeight )

        self.set_size_request( self.theWidth, self.theHeight )
        #self.append( gtk.MenuItem() )
        #self.set_size_request( 150, 450 )

    def popup(self, pms, pmi, func, button, time):
        """
        popup
           - shows this popup memu
        
        return -> None
        This method can throw an exception.
        """
        # shows this popup memu
        gtk.Menu.popup(self, pms, pmi, func, button, time)
        self.show_all()
