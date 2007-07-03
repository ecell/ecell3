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

import gtk
import gobject
import operator
import string
import re
import copy

import ecell.util as util

from constants import *
from PluginInstanceSelection import *
from FullPNQueue import FullPNQueue
from SeparativePane import SeparativePane
from utils import retrieveValueFromListStore, showPopupMessage

DEFAULT_PLUGIN = 'TracerWindow'

class PluginWindowMenu( gtk.Menu ):
    def handleCreatePluginWindow( self, widget, aPluginWindowType ):
        self.emit(
            'ecell-ui-osogo-create-plugin-window',
            aPluginWindowType
            )

    def handleCreateLogger( self, widget ):
        self.emit( 'ecell-ui-osogo-create-logger' )

    def handleAddToBoard( self, widget, aPluginWindowType ):
        self.emit(
            'ecell-ui-osogo-add-to-board',
            aPluginWindowType
            )

    def handleAppendDataTo( self, widget, aPluginWindowTitle ):
        self.emit(
            'ecell-ui-osogo-append-data-to',
            aPluginWindowTitle
            )

    def __init__( self, aSession ):
        gtk.Menu.__init__( self )
        self.theSession = aSession

        # creaets menus of PluginWindow
        aModuleList = self.theSession.getLoadedModules()
        for aPluginWindowType in aModuleList:
            aMenuItem = gtk.MenuItem( "Open with %s..." % aPluginWindowType )
            aMenuItem.connect( 'activate', self.handleCreatePluginWindow,
                    aPluginWindowType )
            aMenuItem.set_name( aPluginWindowType )
            if aPluginWindowType == DEFAULT_PLUGIN:
                self.prepend( aMenuItem )
            else:
                self.append( aMenuItem )

        # appends separator
        self.append( gtk.MenuItem() )

        # menus for Logger
        # creates menu of Logger
        aMenuString = "Create Logger"
        aMenuItem = gtk.MenuItem( aMenuString )
        aMenuItem.connect('activate', self.handleCreateLogger )
        aMenuItem.set_name( aMenuString )
        self.append( aMenuItem )

        # appends separator
        self.append( gtk.MenuItem() )

        # menus for Board
        # creates menu of Board
        aSubMenu = gtk.Menu()

        aMenuString = "Add to Board"
        aMenuItem = gtk.MenuItem( aMenuString )
        aMenuItem.set_name( aMenuString )

        for aPluginWindowType in aModuleList:
            aSubMenuItem = gtk.MenuItem( aPluginWindowType )
            aSubMenuItem.connect(
                'activate',
                self.handleAddToBoard,
                aPluginWindowType )
            aSubMenuItem.set_name( aPluginWindowType )
            if aPluginWindowType == DEFAULT_PLUGIN:
                aSubMenu.prepend( aSubMenuItem )
            else:
                aSubMenu.append( aSubMenuItem )

        aMenuItem.set_submenu( aSubMenu )
        aMenuItem.set_sensitive(
            self.theSession.isFundamentalWindowShown( 'BoardWindow' )
            )
        self.append( aMenuItem )

        # appends separator
        self.append( gtk.MenuItem() )

        aMenuString = "Append data to"
        aMenuItem = gtk.MenuItem( aMenuString )

        if len( self.theSession.getPluginInstanceList() ) == 0:
            aMenuItem.set_sensitive( False )
        else:
            # creates submenu
            aSubMenu = gtk.Menu()
            # creaets menus of PluginWindow instances
            aMenuItemFlag = False
            for aPluginInstance in self.theSession.getPluginInstanceList():
                if aPluginInstance.theViewType == MULTIPLE:
                    aTitle = aPluginInstance.getTitle()
                    aSubMenuItem = gtk.MenuItem( aTitle )
                    aSubMenuItem.connect(
                        'activate',
                        self.handleAppendDataTo,
                        aTitle
                        )
                    aSubMenuItem.set_name( aTitle )
                    aSubMenu.append( aSubMenuItem )
                    aMenuItemFlag = True
            if aMenuItemFlag:
                aMenuItem.set_submenu( aSubMenu )
                aMenuItem.set_sensitive( True )
            else:
                aMenuItem.set_sensitive( False )
        self.append( aMenuItem )
        self.connect( 'cancel', lambda x: gobject.idle_add( self.destroy ) )

gobject.type_register(
    PluginWindowMenu,
    'ecell-ui-osogo-EntityListWindow-PluginWindowMenu'
    )

gobject.signal_new(
    'ecell-ui-osogo-create-plugin-window',
    PluginWindowMenu,
    gobject.SIGNAL_RUN_FIRST,
    gobject.TYPE_NONE,
    ( gobject.TYPE_STRING, )
    )

gobject.signal_new(
    'ecell-ui-osogo-create-logger',
    PluginWindowMenu,
    gobject.SIGNAL_RUN_LAST,
    gobject.TYPE_NONE,
    ()
    )

gobject.signal_new(
    'ecell-ui-osogo-add-to-board',
    PluginWindowMenu,
    gobject.SIGNAL_RUN_LAST,
    gobject.TYPE_NONE,
    ( gobject.TYPE_STRING, )
    )

gobject.signal_new(
    'ecell-ui-osogo-append-data-to',
    PluginWindowMenu,
    gobject.SIGNAL_RUN_LAST,
    gobject.TYPE_NONE,
    ( gobject.TYPE_STRING, )
    )

class EntityListWindow( SeparativePane ):
    '''EntityListWindow
    '''

    DEFAULT_VARIABLE_PROPERTY = 'Value'
    DEFAULT_PROCESS_PROPERTY = 'Activity'

    COMMON_COLUMN_INFO_MAP= {
        'ID':        gobject.TYPE_STRING,
        'Classname': gobject.TYPE_STRING,
        'Path':      gobject.TYPE_STRING
        }

    VARIABLE_COLUMN_INFO_MAP= {
        # Name:      Type

        'Value':     gobject.TYPE_STRING
        }

    PROCESS_COLUMN_INFO_MAP= {
        'Activity':  gobject.TYPE_STRING
        }

    VARIABLE_COLUMN_LIST = [ 'ID', 'Value', 'Classname', 'Path' ]
    PROCESS_COLUMN_LIST = [ 'ID', 'Activity', 'Classname', 'Path' ]


    def __init__( self ):
        '''
        Constructor
        '''
        # call superclass's constructor 
        SeparativePane.__init__( self )

        # initialize parameters
        self.theSelectedFullPNList = []

        self.searchString = ''

        self.thePropertyWindow = None
        self.thePluginInstanceSelection = None
        self.theSession = None

    def setSession( self, aSession ):
        self.theSession = aSession

    def initUI( self ):
        # call superclass's initUI
        SeparativePane.initUI( self )

        self['search_method'].set_property('active', 0)
        self['search_scope'].set_property('active', 0)
        
        # add handers
        self.addHandlers( { 
            'on_system_tree_button_press_event' : self.onEntityClicked,
            'on_view_button_clicked':
                lambda w: self.createPluginWindow(
                    self.getActiveViewPluginName() ),
            'on_variable_tree_button_press_event': self.onEntityClicked,
            'on_process_tree_button_press_event': self.onEntityClicked,
            # search 
            'on_search_button_clicked': self.pushSearchButton,
            'on_search_entry_key_press_event':
            self.keypressOnSearchEntry,
            'on_clear_button_clicked': self.pushClearButton, 
            'on_search_scope_changed': self.searchScopeChanged
            } )

        self.entitySelected = False
        self.theLastSelectedWindow = None
        self.thePopupMenu = None

        self.donotHandle = False
        self.systemTree   = self['system_tree']
        self.processTree  = self['process_tree']
        self.variableTree = self['variable_tree']

        # initialize components
        self.populateSystemTree()
        self.systemTreeConstructed = False
        if self.theSession.getModelWalker() != None:
            self.reconstructSystemTree()
 
        self.populateProcessTree()
        self.populateVariableTree()
        self.populatePluginSelectionMenu()

        selection = self.systemTree.get_selection()
        selection.set_mode( gtk.SELECTION_MULTIPLE )
        selection.connect( 'changed', self.selectSystem )
        selection = self.processTree.get_selection()
        selection.set_mode( gtk.SELECTION_MULTIPLE )
        selection.connect( 'changed', self.selectProcess )
        selection = self.variableTree.get_selection()
        selection.set_mode( gtk.SELECTION_MULTIPLE )
        selection.connect( 'changed', self.selectVariable )

        aFullPN = util.convertFullIDToFullPN( util.createFullID ( 'System::/' ) )
        self.theQueue = FullPNQueue( [ aFullPN ] )
        self.addChild( self.theQueue, 'navigator_area' )
        self.theQueue.initUI()

        self.theQueue.registerCallback( self.doSelection )

        # property window
        aPropertyWindow = self.theSession.createPluginWindow(
            'PropertyWindow',
            [ ( SYSTEM, '', '/', '' ) ] )
        aPropertyWindow.setStatusBar( self.theSession.getStatusBar() )
        self.addChild( aPropertyWindow, 'property_area' )
        aPropertyWindow.initUI()

        self.thePropertyWindow = aPropertyWindow

        # initialize buffer
        self.theSelectedEntityList = []
        self.theSelectedPluginInstanceList = []

        # initialize Add to Board button
        self.theCloseOrder = False
        self.setButtonState( self.theSession.isModelLoaded() )

    def handleSessionEvent( self, event ):
        if event.type == 'model_loaded':
            self.setButtonState( True )

    def setButtonState( self, aState ):
        self['search_button'].set_sensitive( aState )
        self['view_button'].set_sensitive( aState )
        self['search_entry'].set_sensitive( aState )
        self['plugin_optionmenu'].set_sensitive( aState )

    def getQueue( self ):
        return self.theQueue

    def getActiveViewPluginName( self ):
        return retrieveValueFromListStore(
            self['plugin_optionmenu'].get_model(),
            self['plugin_optionmenu'].get_active(), 1 )

    def destroy( self ):
        if self.theCloseOrder:
            return
        self.theCloseOrder = True

        if self.thePluginInstanceSelection != None:
            self.thePluginInstanceSelection.deleted()
            self.thePluginInstanceSelection = None

        self.theSession.deleteEntityListWindow( self )
        SeparativePane.destroy(self)

    def deletePluginInstanceSelection( self, *arg ):
        """sets 'delete_event' as 'hide_event'
        """

        # hide this window
        self['PluginInstanceSelection'].hide_all()

        # set 'delete_event' uneffective
        return True

    def populateSystemTree( self ):
        """initialize SystemTree
        """
        self.lastSelectedSystem = ""
        treeStore = gtk.TreeStore( gobject.TYPE_STRING )
        self.theSysTreeStore = treeStore
        column = gtk.TreeViewColumn( 'System Tree',
                                     gtk.CellRendererText(),
                                     text=0 )
        column.set_visible( True )
        self.systemTree.append_column(column)

        self.systemTree.set_model( treeStore )

        self.processTree.set_search_column( 0 )
        self.theSysSelection =  self.systemTree.get_selection()

    def populateProcessTree( self ):
        """initialize ProcessTree
        """

        columnTypeList = []

        for i in range( len( self.PROCESS_COLUMN_LIST ) ):
            title = self.PROCESS_COLUMN_LIST[i]

            try:
                type = self.PROCESS_COLUMN_INFO_MAP[ title ]
            except:
                type = self.COMMON_COLUMN_INFO_MAP[ title ]

            column = gtk.TreeViewColumn( title,
                                         gtk.CellRendererText(),
                                         text=i )
            column.set_reorderable( True )
            column.set_sort_column_id( i )
            self.processTree.append_column( column )
            columnTypeList.append( type )
            if type == gobject.TYPE_FLOAT:
                column.set_alignment( 1.0 )
                column.get_cell_renderers()[0].set_property( 'xalign', 1.0 )

        self.processTree.set_search_column( 0 )

        model = gtk.ListStore( *columnTypeList )
        self.processTree.set_model( model )

    def populateVariableTree( self ):
        """initializes VariableTree
        """

        columnTypeList = []

        for i in range( len( self.VARIABLE_COLUMN_LIST ) ):
            title = self.VARIABLE_COLUMN_LIST[i]

            try:
                type = self.VARIABLE_COLUMN_INFO_MAP[ title ]
            except:
                type = self.COMMON_COLUMN_INFO_MAP[ title ]

            column = gtk.TreeViewColumn( title,
                                         gtk.CellRendererText(),
                                         text=i )
            column.set_reorderable( True )
            column.set_sort_column_id( i )
            self.variableTree.append_column( column )
            columnTypeList.append( type )
            if type == gobject.TYPE_FLOAT:
                column.set_alignment( 1.0 )
                column.get_cell_renderers()[0].set_property( 'xalign', 1.0 )

        self.variableTree.set_search_column( 0 )

        model = gtk.ListStore( *columnTypeList )
        self.variableTree.set_model( model )

    def populatePluginSelectionMenu( self ):
        """initializes PluginWindowOptionMenu
        """
        anOptionMenu = self['plugin_optionmenu']
        aRenderer = gtk.CellRendererText()
        aList = gtk.ListStore( gobject.TYPE_STRING, gobject.TYPE_STRING )
        for aPluginWindowName in self.theSession.getLoadedModules():
            if aPluginWindowName == DEFAULT_PLUGIN:
                anIter = aList.prepend()
            else:
                anIter = aList.append()
            aList.set( anIter, 0, aPluginWindowName, 1, aPluginWindowName )
        anOptionMenu.set_model( aList )
        anOptionMenu.pack_start( aRenderer, True )
        anOptionMenu.set_attributes( aRenderer, text = 0 )
        anOptionMenu.set_active( 0 )
        anOptionMenu.show_all()

    def __openPluginInstanceSelectionWindow( self, *arg ):
        """open PluginInstanceSelectionWindow
        Returns None
        """

        if self.thePluginInstanceSelection != None:
            self.thePluginInstanceSelection.present()
        else:
            self.thePluginInstanceSelection = \
                PluginInstanceSelection( self.theSession, self )
            self.thePluginInstanceSelection.initUI()

            # updates list of PluginInstance
            self.thePluginInstanceSelection.update()

    def __updatePluginInstanceSelectionWindow2( self ):
        """updates list of PluginInstanceSelectionWindow
        Returns None
        """

        self.thePluginInstanceListStore.clear()
        for aPluginInstance in self.theSession.getPluginInstanceList():
            if aPluginInstance.theViewType == MULTIPLE:
                aPluginInstanceTitle = aPluginInstance.getTitle()
                iter = self.thePluginInstanceListStore.append()
                self.thePluginInstanceListStore.set_value( iter, 0, aPluginInstanceTitle )
                self.thePluginInstanceListStore.set_data( aPluginInstanceTitle, aPluginInstanceTitle )

    def closePluginInstanceSelectionWindow( self, *arg ):
        """closes PluginInstanceSelectionWindow
        Returns None
        """

        if self.thePluginInstanceSelection != None:
            #self.thePluginInstanceSelection['PluginInstanceSelection'].hide_all()
            self.thePluginInstanceSelection.deleted()
            self.thePluginInstanceSelection = None

    def onEntityClicked( self, aWidget, anEvent ):
        """displays popup menu only when right button is pressed.
        aWidget   --  EntityListWindow
        anEvent   --  an event
        Returns None
        [Note]:creates and adds submenu that includes menus of PluginWindow instances
        """
        # When the user double-clicked the left button
        if anEvent.type == gtk.gdk._2BUTTON_PRESS:
            aSelectedRawFullPNList = self.__getSelectedRawFullPNList()
            self.createPluginWindow( self.getActiveViewPluginName() )
        # When the user hit the right button
        if anEvent.type == gtk.gdk.BUTTON_PRESS and anEvent.button == 3:
            # displays all items on PopupMenu
            aPopupMenu = PluginWindowMenu( self.theSession )
            aPopupMenu.connect(
                'ecell-ui-osogo-create-plugin-window',
                lambda w, aWindowType: self.createPluginWindow( aWindowType )
                )
            aPopupMenu.connect(
                'ecell-ui-osogo-create-logger',
                lambda w: self.createLogger()
                )
            aPopupMenu.connect(
                'ecell-ui-osogo-add-to-board',
                lambda w, aPluginWindowType:
                    self.addToBoard( aPluginWindowType )
                )
            aPopupMenu.connect(
                'ecell-ui-osogo-append-data-to',
                lambda w, aPluginWindowTitle:
                    self.appendSelected( aPluginWindowTitle )
                )
            aPopupMenu.show_all() 
            # displays popup menu
            aPopupMenu.popup(None,None,None,anEvent.button,anEvent.time)

    def update( self ):
        """overwrite superclass's method
        updates this window and property window
        Returns None
        """
        if not self.exists():
            return
        if self.theSession.getModelWalker() == None:
            return

        elif not self.systemTreeConstructed:
            self.reconstructSystemTree()

        # updates property window
        self.thePropertyWindow.update()

        # update PluginInstanceSelectionWindow
        if self.thePluginInstanceSelection != None:
            self.thePluginInstanceSelection.update()

        self.updateLists()

    def constructSystemTree( self, parent, fullID ):
        newlabel = fullID[ID] 

        systemStore = self.systemTree.get_model()
        iter  = systemStore.append( parent )
        systemStore.set_value( iter, 0, newlabel )
        key = str( systemStore.get_path( iter ) )
        systemStore.set_data( key, fullID )

        systemPath = util.createSystemPathFromFullID( fullID )
        systemList = self.theSession.getEntityList( 'System', systemPath )
        systemListLength = len( systemList )

        if  systemListLength == 0:
            return

        for systemID in systemList:
            newSystemFullID = ( SYSTEM, systemPath, systemID )
            self.constructSystemTree( iter, newSystemFullID )

            path = systemStore.get_path( iter )
            if systemListLength < 6 and len( path ) < 6:
                self.systemTree.expand_row( path, True )

    def reconstructSystemTree( self ):
        rootSystemFullID = util.createFullID( 'System::/' )
        self.constructSystemTree( None, rootSystemFullID )
        self.systemTreeConstructed = True

    def reconstructLists( self ):
        selectedSystemList = self.getSelectedSystemList()

        if len( selectedSystemList ) == 0:
            return
        if self.entitySelected:
            return 
        # Variable list
        self.reconstructEntityList(
            'Variable', self.variableTree,
            selectedSystemList,
            self.VARIABLE_COLUMN_LIST,
            self.VARIABLE_COLUMN_INFO_MAP )

        # Process list
        self.reconstructEntityList(
            'Process', self.processTree,
            selectedSystemList,
            self.PROCESS_COLUMN_LIST,
            self.PROCESS_COLUMN_INFO_MAP )
        self.updateListLabels()

    def reconstructEntityList( self, type, view, systemList, columnList, columnInfoList ):
        # get the entity list in the selected system(s)
        typeID = ENTITYTYPE_DICT[ type ]

        fullIDList = []
        for systemFullID in systemList:

            systemPath = util.createSystemPathFromFullID( systemFullID )

            idList = self.theSession.getEntityList( type, systemPath )
            fullIDList += [ ( typeID, systemPath, id ) for id in idList ]


        entityStore = view.get_model()

        # clear the store
        donotHandle = self.donotHandle
        self.donotHandle = True
        entityStore.clear()
        self.donotHandle = donotHandle

        #		columnList = view.get_columns()
        # re-create the list
        for fullID in fullIDList:

            ID = fullID[2]
            # temporary hack for the entity searching.
            # this can be like this in python 2.3 or above:
            #    if not self.searchString in ID:
            if ID.find( self.searchString ) < 0:
                continue

            stub = self.theSession.createEntityStub( fullID )

            valueList = []

            for title in columnList:

                if title in self.COMMON_COLUMN_INFO_MAP.keys():
                    if title == 'ID':
                        value = ID
                    elif title == 'Classname':
                        value = stub.getClassname()
                    elif title == 'Path':
                        value =  fullID [SYSTEMPATH]
                    else:
                        raise "Unexpected error: invalid column title."
                else:
                    value = stub[ title ] # if not common, it's entity property

                valueList.append( value )

            iter = entityStore.append( valueList )
            iterString = entityStore.get_string_from_iter( iter )
            entityStore.set_data( iterString, util.createFullIDString( fullID ) )

    def doSelection( self, aFullPNList ):
        if self.theSession.getModelWalker() == None:
            return
        if len( aFullPNList ) == 0:
            return
        self.doSelectSystem( aFullPNList ) 
        self.doSelectProcess( aFullPNList )
        self.doSelectVariable( aFullPNList )

    def doSelectSystem( self, aFullPNList ):
        targetFullIDList = []

        if aFullPNList[0][TYPE] != SYSTEM:
            targetFullIDList += [ util.createFullIDFromSystemPath(
                aFullPN[SYSTEMPATH] )  for aFullPN in aFullPNList ]
        else:
            for aFullPN in aFullPNList:
                targetFullIDList.append( util.convertFullPNToFullID( aFullPN ) )

        # if to slow there should be a check whether this is needed in all cases
        donotHandle = self.donotHandle
        self.donotHandle = True
        self.theSysSelection.unselect_all()
        self.theSysSelection.set_mode( gtk.SELECTION_MULTIPLE )

        for  targetFullID in targetFullIDList:
            #doselection
            targetPath = util.createSystemPathFromFullID( targetFullID )
            anIter = self.getSysTreeIter( targetPath )

            aPath = self.theSysTreeStore.get_path( anIter )
            self.__expandRow( aPath )
            self.theSysSelection.select_iter( anIter )

        self.donotHandle = donotHandle

        self.reconstructLists()

    def getSysTreeIter( self, aSysPath, anIter = None ):
        """
        returns iter of string aSysPath or None if not available
        """
        systemStore = self.systemTree.get_model()
        if anIter == None:
            anIter = systemStore.get_iter_first()
            if aSysPath == '/':
                return anIter
            else:
                aSysPath = aSysPath.strip ('/')

        # get first path string
        anIndex = aSysPath.find( '/' )
        if anIndex == -1:
            anIndex = len( aSysPath )
        firstTag = aSysPath[ 0 : anIndex ]

        # create remaining path string
        aRemainder = aSysPath[ anIndex + 1 : len( aSysPath ) ]

        # find iter of first path string
        numChildren = systemStore.iter_n_children( anIter )
        isFound = False
        for i in range( 0, numChildren):
            childIter = systemStore.iter_nth_child( anIter, i )

            if systemStore.get_value( childIter, 0) == firstTag:
                isFound = True
                break

        # if not found return None
        if not isFound:
            return None

        # if remainder is '' return iter
        if aRemainder == '':
            return childIter

        # return recursive remainder with iter
        return self.getSysTreeIter( aRemainder, childIter )

    def __expandRow( self, aPath ):
        """
        in: gtktreePath aPath
        """
        if not self.systemTree.row_expanded( aPath ):

            # get iter
            anIter = self.theSysTreeStore.get_iter( aPath)

            # get parent iter
            parentIter = self.theSysTreeStore.iter_parent( anIter )

            # if iter is root expand
            if parentIter != None:

                # if not get parent path
                parentPath = self.theSysTreeStore.get_path( parentIter )
                
                # expand parentpath
                self.__expandRow( parentPath )
                
            # expand this path
            self.systemTree.expand_row( aPath, False )

    def selectListIter( self, aListStore, aSelection, anIDList ):
        anIter = aListStore.get_iter_first()
        while anIter != None:
            if aListStore.get_value( anIter, 0 ) in anIDList:
                donotHandle = self.donotHandle
                self.donotHandle = True
                aSelection.select_iter( anIter )
                self.donotHandle = donotHandle
            anIter = aListStore.iter_next( anIter )

    def doSelectProcess( self, aFullPNList ):
        #unselect all

        selection = self.processTree.get_selection()
        listStore = self.processTree.get_model()
        selection.unselect_all()
        selection.set_mode(gtk.SELECTION_MULTIPLE )

        if aFullPNList[0][TYPE] == PROCESS:
            # do select
            self.selectListIter( listStore, selection, self.__createIDList( aFullPNList ) )
    
    def __createIDList( self, aFullPNList ):
        anIDList = []
        for aFullPN in aFullPNList:
            anIDList.append( aFullPN[ID] )
        return anIDList
    
    def doSelectVariable( self, aFullPNList ):
        #unselect all
        selection = self.variableTree.get_selection()
        listStore = self.variableTree.get_model()
        selection.unselect_all()
        selection.set_mode(gtk.SELECTION_MULTIPLE )

        if aFullPNList[0][TYPE] == VARIABLE:
            # do select
            self.selectListIter( listStore, selection, self.__createIDList( aFullPNList ) )

    def selectSystem( self, obj ):
        if self.donotHandle:
            return

        # select the first selected System in the PropertyWindow
        systemFullIDList = self.getSelectedSystemList()
        fullPNList = []
        for systemFullID in systemFullIDList:
            fullPNList.append( util.convertFullIDToFullPN( systemFullID ) )
        self.donotHandle = True
        self.theQueue.pushFullPNList( fullPNList )
        self.donotHandle = False

    def getSelectedSystemList( self ):
        '''
        Return - a list of FullIDs of the currently selected Systems.
        '''

        # get system ID from selected items of system tree,
        # and get entity list from session

        systemList = []

        selection = self.systemTree.get_selection()
        selectedSystemTreePathList = selection.get_selected_rows()
        if selectedSystemTreePathList == None:
            return []
        selectedSystemTreePathList = selectedSystemTreePathList[1]
        
        systemStore = self.systemTree.get_model()

        for treePath in selectedSystemTreePathList:
            systemFullID = systemStore.get_data( str( treePath ) )
            systemList.append( systemFullID )

        return systemList

    def updateLists( self ):
        '''
        This method updates property values shown in the list of
        Variables and Processes.
        '''

        self.updateEntityList(
            'Process', self.processTree.get_model(),
            self.PROCESS_COLUMN_LIST,
            self.PROCESS_COLUMN_INFO_MAP )

        self.updateEntityList(
            'Variable', self.variableTree.get_model(),
            self.VARIABLE_COLUMN_LIST,
            self.VARIABLE_COLUMN_INFO_MAP )

    def updateEntityList( self, type, model, columnList, columnInfoMap ): 
        propertyColumnList= [
            ( columnList.index( i ), i )
            for i in columnInfoMap.keys() ]
        for row in model:
            iter = row.iter
            aFullIDString = model.get_data( model.get_string_from_iter( iter ) )

            stub = self.theSession.createEntityStub(
                util.createFullID( aFullIDString ) )

            columnList = []
            for propertyColumn in propertyColumnList:
                newValue = stub[ propertyColumn[1] ]
                columnList += [ propertyColumn[0], "%g"%(newValue) ] 

            model.set( iter, *columnList )

    def updateListLabels( self ):

        self.__updateViewLabel( 'Variable', self['variable_label'],\
                                self.variableTree )
        self.__updateViewLabel( 'Process', self['process_label'],\
                                self.processTree )
        self.__updateViewLabel( 'System', self['system_label'],\
                                self.systemTree )

    def __updateViewLabel( self, type, label, view ):
        shownCount    = len( view.get_model() )
        selectedCount = view.get_selection().count_selected_rows()
        labelText = type + ' (' + str( selectedCount ) + '/' + \
                    str( shownCount ) + ')' 
        label.set_text( labelText )

    def selectProcess( self, selection ):
        if self.donotHandle:
            return
        self.entitySelected = True
        self.theLastSelectedWindow = "Process"

        # clear fullPN list
        self.theSelectedFullPNList = []

        # get selected items
        selection.selected_foreach(self.process_select_func)

        if len(self.theSelectedFullPNList)>0:
            self.donotHandle = True
            self.theQueue.pushFullPNList( self.theSelectedFullPNList )
            self.donotHandle = False
        self.updateListLabels()
        self.entitySelected = False
        
    def selectVariable( self, selection ):
        if self.donotHandle:
            return
        self.entitySelected = True
        self.theLastSelectedWindow = "Variable"

        # clear fullPN list
        self.theSelectedFullPNList = []

        # get selected items
        selection.selected_foreach(self.variable_select_func)

        if len(self.theSelectedFullPNList)>0:
            self.donotHandle = True
            self.theQueue.pushFullPNList( self.theSelectedFullPNList )
            self.donotHandle = False

        # clear selection of process list
#        self.processTree.get_selection().unselect_all()

        self.updateListLabels()
        self.entitySelected = False

    def variable_select_func(self,tree,path,iter):
        '''function for variable list selection

        Return None
        '''

        model = self.variableTree.get_model()
        data = model.get_data( model.get_string_from_iter( iter ) )
        assert( data != None )
        entityFullID = util.createFullID( data )
        entityFullPN = entityFullID + ( self.DEFAULT_VARIABLE_PROPERTY, )
        self.theSelectedFullPNList.append( entityFullPN )

    def process_select_func(self,tree,path,iter):
        '''function for process list selection

        Return None
        '''
        model = self.processTree.get_model()
        data = model.get_data( model.get_string_from_iter( iter ) )
        entityFullID = util.createFullID( data )
        entityFullPN = entityFullID + ( self.DEFAULT_PROCESS_PROPERTY, )
        self.theSelectedFullPNList.append( entityFullPN )

    def createPluginWindow( self, aPluginWindowType ) :
        """creates new PluginWindow instance(s)
        *obj   --  gtk.MenuItem on onEntityClicked or gtk.Button
        Returns None
        """
        self.thePropertyWindow.clearStatusBar()

        aSelectedRawFullPNList = self.__getSelectedRawFullPNList()

        # When no FullPN is selected, displays error message.
        if aSelectedRawFullPNList  == None \
           or len( aSelectedRawFullPNList ) == 0:
            aMessage = 'No entity is selected.'
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            return False

        self.theSession.openPluginWindow(
            aPluginWindowType, self.thePropertyWindow.getFullPNList() )

    def appendSelected( self, aTitle ):
        """appends RawFullPN to PluginWindow instance
        Returns True(when appened) / False(when not appened)
        """

        # clear status bar
        self.thePropertyWindow.clearStatusBar()

        aMessage = None

        anInstance = self.theSession.findPluginInstanceByTitle( aTitle )
        try:
            anInstance.appendRawFullPNList( self.__getSelectedRawFullPNList() )
            aMessage = "Selected Data are added to %s" % aTitle
        except Exception, e:
            aMessage = "Can't append data to %s" % aTitle

        if aMessage != None:
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            self.theSession.message(aMessage)

    def appendSelectedMultiple( self ):
        self.theSelectedPluginInstanceList = []
        selection = self.thePluginInstanceSelection['plugin_tree'].get_selection()
        selection.selected_foreach(self.thePluginInstanceSelection.plugin_select_func)

        # When no FullPN is selected, displays error message.
        if self.__getSelectedRawFullPNList() == None or len( self.__getSelectedRawFullPNList() ) == 0:
            showMessage(OK_MODE, "No entity selected", 'Error')
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            return False

        # When no plugin instance is selected, displays error message.
        if len(self.theSelectedPluginInstanceList) == 0:

            showMessage(OK_MODE, "No plugin selected", 'Error')
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            return False

        # buffer of appended instance's title
        anAppendedTitle = []

        anErrorFlag = False

        # appends data
        for aPluginWindowTitle in self.theSelectedPluginInstanceList:
            for anInstance in self.theSession.getPluginInstanceList():
                if anInstance.getTitle() == aPluginWindowTitle:
                    try:
                        anInstance.appendRawFullPNList( self.__getSelectedRawFullPNList() )
                    except TypeError:
                        anErrorFlag = True
                        aMessage = "Can't append data to %s" %str(anInstance.getTitle())
                        self.thePropertyWindow.showMessageOnStatusBar(aMessage)
                    else:
                        anAppendedTitle.append( anInstance.getTitle() )
                    break

        # When at least one instance is appended
        if len( anAppendedTitle ) <= 0 or anErrorFlag == True:
            return False

        # displays message
        aMessage = "Selected Data are added to %s" %str(anAppendedTitle)
        self.theSession.message(aMessage)
        self.thePropertyWindow.showMessageOnStatusBar(aMessage)

        # closes PluginInstanceSelectionWindow
        self.closePluginInstanceSelectionWindow()
        return True

    def __getSelectedRawFullPNList( self ):
        """
        Return a list of selected FullPNs
        """
        return self.theQueue.getActualFullPNList()
        
        # this is redundant
        self.theSelectedFullPNList = []

        if ( self.theLastSelectedWindow == "None" ):
            return None

        if ( self.theLastSelectedWindow == "Variable" ):

            selection=self.variableTree.get_selection()
            selection.selected_foreach(self.variable_select_func)

        if ( self.theLastSelectedWindow == "Process" ):

            selection=self.processTree.get_selection()
            selection.selected_foreach(self.process_select_func)

        if len(self.theSelectedFullPNList) == 0:
            selectedSystemList = self.getSelectedSystemList()
            for aSystemFullID in selectedSystemList:
                self.theSelectedFullPNList.append( util.convertFullIDToFullPN( aSystemFullID ) )


        # If no property is selected on PropertyWindow, 
        # create plugin Window with default property (aValue) 
        if len( str(self.thePropertyWindow.getSelectedFullPN()) ) == 0:
            return self.theSelectedFullPNList

        # If a property is selected on PropertyWindow, 
        # create plugin Window with selected property
        else:
            return [self.thePropertyWindow.getSelectedFullPN()]

    def addToBoard( self, aPluginWindowType ):
        """add plugin window to board
        """

        self.thePropertyWindow.clearStatusBar()

        aSelectedRawFullPNList = self.__getSelectedRawFullPNList()

        # When no FullPN is selected, displays error message.
        if aSelectedRawFullPNList  == None or \
           len( aSelectedRawFullPNList ) == 0:
            aMessage = 'No entity is selected.'
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            return False

        self.theSession.getFundamentalWindow('BoardWindow').addPluginWindows(
            aPluginWindowType, self.__getSelectedRawFullPNList() )

    def createLogger( self ):
        """creates Logger about all FullPN selected on EntityTreeView
        Returns None
        """

        # clear status bar
        self.thePropertyWindow.clearStatusBar()

        # gets selected RawFullPNList
        aSelectedRawFullPNList = self.__getSelectedRawFullPNList()


        # When no entity is selected, displays confirm window.
        if len(aSelectedRawFullPNList) == 0:


            aMessage = 'No Entity is selected.'
            self.thePropertyWindow.showMessageOnStatusBar(aMessage)
            showPopupMessage( OK_MODE, aMessage, 'Error' )
            return None

        # creates Logger using PropertyWindow
        #self.theQueue.pushFullPNList( aSelectedRawFullPNList )
        self.thePropertyWindow.createLogger()

        # display message on status bar
        if len(aSelectedRawFullPNList) == 1:
            aMessage = 'Logger was created.'
        else:
            aMessage = 'Loggers were created.'
        self.thePropertyWindow.showMessageOnStatusBar(aMessage)
        #self.checkCreateLoggerButton()

    def searchEntity( self ):
        """search Entities
        Returns None
        """
        searchString = self['search_entry'].get_text()
        # set modelwalker to current selection
        aFullPNList = []

        self.theSession.getModelWalker().reset()
        
        nextFullID = self.theSession.getModelWalker().getCurrentFullID()

        while nextFullID != None:
            if nextFullID[TYPE] == SYSTEM:
                currentSystemID = nextFullID
                currentSystemSelected = False
                
            elif not currentSystemSelected and nextFullID[ID].find( searchString ) != -1:
                # select
                aFullPNList += [ util.convertFullIDToFullPN( currentSystemID ) ]
                currentSystemSelected = True
                
            nextFullID = self.theSession.getModelWalker().getNextFullID()
            

        if len( aFullPNList ) == 0:
            showPopupMessage(
                OK_MODE, 
                "Search string %s not found." % searchString, 
                "Search failed" )
            return
        self.searchString = searchString
        self.theQueue.pushFullPNList( aFullPNList )
        self['search_button'].set_sensitive( False)
        if self.searchString != '':
            self['clear_button'].set_sensitive(True )

    def filterSelectedSystems( self ):
        self.searchString = self['search_entry'].get_text()
        self.reconstructLists()
        self['search_button'].set_sensitive( False)
        if self.searchString != '':
            self['clear_button'].set_sensitive(True )
        else:
            self['clear_button'].set_sensitive(False )

    def pushSearchButton( self, *arg ):
        searchScope = self['search_scope'].get_property( 'active' )
        if searchScope == 0:
            self.searchEntity()
        else:
            self.filterSelectedSystems()

    def keypressOnSearchEntry( self, *arg ):
        if( arg[1].keyval == 65293 ):

            self.pushSearchButton( None )
        else :
            self['search_button'].set_sensitive(True )
        
    def pushClearButton( self, *args ):
        self['search_entry'].set_text('')
        self.filterSelectedSystems()
        
    def searchScopeChanged( self, *args ):
        searchString = self['search_entry'].get_text()
        if self.searchString != '':
            self['search_button'].set_sensitive( True)
        else:
            self['search_button'].set_sensitive( False)

