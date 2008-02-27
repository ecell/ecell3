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
import operator

import gobject
import gtk
import gtk.gdk

import ecell.util as util

from Pane import Pane
from utils import *

VARREF_NAME = 0
VARREF_FULLID = 1
VARREF_COEF = 2

NAME_COLUMN = 0
FULLID_COLUMN = 1
COEF_COLUMN = 2
EDITABLE_COLUMN = 3

class VariableReferenceEditor( Pane ):
    columnSpecList = [
        {
            'type': gobject.TYPE_STRING,
            'column': NAME_COLUMN,
            'visible': True,
            'label': 'Name',
            },
        {
            'type': gobject.TYPE_STRING,
            'column': NAME_COLUMN,
            'visible': True,
            'label': 'FullID',
            },
        {
            'type': gobject.TYPE_STRING,
            'column': FULLID_COLUMN,
            'visible': True,
            'label': 'Coefficient',
            },
        {
            'type': gobject.TYPE_BOOLEAN,
            'column': EDITABLE_COLUMN,
            'visible': False,
            'label': '',
            },
        ]

    def __init__( self ):
        Pane.__init__( self )
        self.theFullID = None

    def initUI( self ):
        Pane.initUI( self )
        self.theTreeView = self.theRootWidget
        self.theListStore = gtk.ListStore(
            *map( lambda x: x['type'], self.columnSpecList ) )
        self.theTreeView.set_model( self.theListStore )
        for columnSpec in self.columnSpecList:
            rdr = gtk.CellRendererText()
            rdr.connect(
                'edited', self.__cellEdited, columnSpec['column'])
            col = gtk.TreeViewColumn(
                columnSpec['label'], rdr,
                text = columnSpec['column'], editable = EDITABLE_COLUMN )
            col.set_visible( columnSpec['visible'] )
            col.set_resizable( True )
            col.set_reorderable( True )
            col.set_sort_column_id = columnSpec['column']
            self.theTreeView.append_column( col )
        self.theTreeView.size_allocate( gtk.gdk.Rectangle( 0, 0, 100, 1000 ) )
        self.theTreeView.connect( "button-press-event", self.__buttonPressed )

    def setDisplayedFullID ( self, aFullID ):
        self.theFullID = aFullID
        self.update()
        
    def getProcessFullID( self ):
        return self.theFullID
        
    def update( self ):
        # gets varreflist
        theValue = self.theParent.theSession.getEntityProperty(
                self.theFullID.createFullPN( 'VariableReferenceList' ) )
        
        #redraw whole list
        self.theListStore.clear()
        anIter = None

        for aVariableReference in theValue:
            aName = aVariableReference[VARREF_NAME]
            aFullIDString = aVariableReference[VARREF_FULLID]
            aCoef = aVariableReference[VARREF_COEF]
            anIter = self.theListStore.append()
            # to make columns editable change False to True 
            self.theListStore.set(
                anIter,
                NAME_COLUMN, aName,
                FULLID_COLUMN, aFullIDString,
                COEF_COLUMN, aCoef,
                EDITABLE_COLUMN, False )
        
    def __setValue( self ):
        #take value from list and redraw
        aVarrefList = []
        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            aVarref = ["","",1, 1]
            aVarref[VARREF_NAME] = self.theListStore.get_value( anIter, NAME_COLUMN )
            aVarref[VARREF_FULLID] = self.theListStore.get_value( anIter, FULLID_COLUMN )
            aVarref[VARREF_COEF] = self.theListStore.get_value( anIter, COEF_COLUMN )
            aVarrefList.append( tuple( aVarref ) )
            anIter = self.theListStore.iter_next( anIter )
        aVarrefListTuple = tuple( aVarrefList )

        try:
            self.theParent.theSession.setEntityProperty(
                util.convertFullIDToFullPN(
                    self.theFullID, 'VariableReferenceList' ),
                aVarrefListTuple )
        except:
            # print out traceback
            import sys
            import traceback
            anErrorMessage = string.join(
                traceback.format_exception( 
                    sys.exc_type,sys.exc_value,sys.exc_traceback ),
                                            '\n' )
            self.theParent.theSession.message(anErrorMessage)
    
            # creates and display error message dialog.
            anErrorMessage = "Could not change variable references"
            showPopupMessage( OK_MODE, anErrorMessage, "Error" )
            self.update()
        
    def __cellEdited( self, *args ):
        aNewValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        column = args[3]
        if column == COEF_COLUMN:
            #check whether it is integer
            if not operator.isNumberType( aNewValue):
                showPopupMessage( OK_MODE,
                    "Coefficient should be numeric.", "Error" )
                return
        self.theListStore.set_value( anIter, column, aNewValue )
        self.__setValue()
        
    def __buttonPressed( self, widget, event ):
        if event.type == gtk.gdk._2BUTTON_PRESS:
            realFullID = self.__getRealFullID()
            if realFullID != None:
                self.__openAction( None, realFullID )
        if event.button == 3:
            self.__popUpMenu()
            return True

    def __getRealFullID( self ):
        selectedFullID = self.__getSelectedFullID()
        if selectedFullID == None:
            return None
        isFullIDReal = self.theParent.theSession.getEntityClassName(
            selectedFullID )
        return isFullIDReal != None and selectedFullID or None

    def __popUpMenu(self ):
        selectedIter = self.__getSelectedIter()
        realFullID = self.__getRealFullID()
        aMenu = gtk.Menu()
        openItem = gtk.MenuItem( "Open" )
        if realFullID != None:
            openItem.connect( "activate", self.__openAction, realFullID )
        else:
            openItem.set_sensitive( False )
        aMenu.append( openItem )

        openNewItem = gtk.MenuItem( "Open in new" )
        if realFullID != None:
            openNewItem.connect( "activate", self.__openNewAction, realFullID )
        else:
            openNewItem.set_sensitive( False )
        aMenu.append( openNewItem )
        aMenu.append( gtk.SeparatorMenuItem() )
        
        addItem = gtk.MenuItem( "Add" )
        addItem.connect("activate", self.__addAction )
        # to add Add functionality uncomment it
        #aMenu.append( addItem )

        deleteItem = gtk.MenuItem( "Delete" )

        if selectedIter != None:
            deleteItem.connect( "activate", self.__deleteAction, selectedIter )
        else:
            deleteItem.set_sensitive( False )
        # to add delete functionality uncomment it
        #aMenu.append( deleteItem )
        aMenu.show_all()
        aMenu.popup( None, None, None, 1, 0 )

    def __openNewAction( self, widget, aFullIDString ):
        theFullPNList =  [
            identifiers.FullPN( identifiers.FullID( aFullIDString ), '' )
            ]
        self.theParent.theSession.openPluginWindow(
            "PropertyWindow", theFullPNList )

    def __openAction ( self, widget, aFullIDString ):
        theFullPNList =  [
            identifiers.FullPN( identifiers.FullID( aFullIDString ), '' )
            ]
        self.theParent.theQueue.pushFullPNList( theFullPNList )
        self.theParent.update( True )
    
    def __addAction ( self, *args ):
        pass
        
    def __deleteAction( self, *args ):
        pass

    def __getSelectedFullID( self ):
        anIter = self.__getSelectedIter()
        if anIter == None:
            return None
        aVarref = identifiers.FullID(
            self.theListStore.get_value( anIter, FULLID_COLUMN ) )
        return self.__getAbsoluteReference( aVarref )
    
    def __getSelectedIter( self ):
        anIter = self.theTreeView.get_selection().get_selected()[1]
        return anIter
        
    def __getAbsoluteReference( self, aFullID ):
        aVariableSystemPath = aFullID.getSuperSystemPath()

        if aVariableSystemPath.isAbsolute():
            aVariableSystemPath = aVariableSystemPath
        else:
            aVariableSystemPath = aVariableSystemPath.toAbsolute(
                self.theFullID.getSuperSystemPath() )
        return identifiers.FullID( VARIABLE, aVariableSystemPath, aFullID.id )
