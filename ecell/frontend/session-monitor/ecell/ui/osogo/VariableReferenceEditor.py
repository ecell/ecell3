#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
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

import operator

import gobject
import gtk
import gtk.gdk

from ecell.ecssupport import *

from ecell.ui.osogo.ConfirmWindow import *

VARREF_NAME = 0
VARREF_FULLID = 1
VARREF_COEF = 2

NAME_COLUMN = 0
FULLID_COLUMN = 1
COEF_COLUMN = 2
EDITABLE_COLUMN = 3
COLOR_COLUMN = 4


class VariableReferenceEditor:
    
    def __init__( self, aParent, attachmentPoint ):
        self.theParent = aParent
        self.theSession = self.theParent.theSession

        self.theListStore = gtk.ListStore( gobject.TYPE_STRING,
                                            gobject.TYPE_STRING,
                                            gobject.TYPE_STRING,
                                            gobject.TYPE_BOOLEAN )
        self.theTreeView = gtk.TreeView()
        self.theTreeView.show()
        self.theTreeView.set_model( self.theListStore )

        rendererName = gtk.CellRendererText( )
        rendererName.connect('edited', self.__cellEdited, NAME_COLUMN)
        columnName=gtk.TreeViewColumn( "Name", rendererName, text=NAME_COLUMN, editable = EDITABLE_COLUMN)
        columnName.set_visible( True )
        columnName.set_resizable( True )
        columnName.set_reorderable( True )
        columnName.set_sort_column_id = NAME_COLUMN
        
        rendererFullID = gtk.CellRendererText()
        rendererFullID.connect('edited', self.__cellEdited, FULLID_COLUMN)
        columnFullID =gtk.TreeViewColumn( "FullID", rendererFullID, text=FULLID_COLUMN, editable=EDITABLE_COLUMN)
        columnFullID.set_visible( True )
        columnFullID.set_resizable( True )
        columnFullID.set_reorderable( True )
        columnFullID.set_sort_column_id ( FULLID_COLUMN )
        
        rendererCoef = gtk.CellRendererText()
        rendererCoef.connect('edited', self.__cellEdited, COEF_COLUMN)
        columnCoef=gtk.TreeViewColumn( "Coefficient", rendererCoef, text=COEF_COLUMN, editable=EDITABLE_COLUMN)
        columnCoef.set_visible( True )
        columnCoef.set_resizable( True )
        columnCoef.set_reorderable( True )
        columnCoef.set_sort_column_id ( COEF_COLUMN )
        
        self.theTreeView.append_column( columnName )
        self.theTreeView.append_column( columnFullID )
        self.theTreeView.append_column( columnCoef )
        self.theTreeView.connect( "button-press-event", self.__buttonPressed )
        attachmentPoint.add( self.theTreeView )
        aFullID, aPropertyName = convertFullPNToFullID( self.theParent.getFullPN() )
        self.setDisplayedFullID ( aFullID )
        
        
    def setDisplayedFullID ( self, aFullID ):
        self.theFullID = aFullID
        self.theFullIDString = createFullIDString( self.theFullID )
        self.update()
        
    def getProcessFullID( self ):
        return self.theFullID
        
    def update( self ):
        # gets varreflist
        theValue = self.theSession.createEntityStub( self.theFullIDString ).getProperty( "VariableReferenceList" )
        #redraw whole list
        self.theListStore.clear()
        anIter = None

        for aVariableReference in theValue:
            aName = aVariableReference[VARREF_NAME]
            aFullID = aVariableReference[VARREF_FULLID]
            aCoef = aVariableReference[VARREF_COEF]
            anIter = self.theListStore.append(  )
            # to make columns editable change False to True 
            self.theListStore.set( anIter, 0, aName, 1, aFullID, 2, aCoef, 3, False )
        
        
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
            self.theSession.createEntityStub( self.theFullIDString ).setProperty( "VariableReferenceList", aVarrefListTuple )
        except:
            # print out traceback
            import sys
            import traceback
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
            self.theSession.message("-----An error happens.-----")
            self.theSession.message(anErrorMessage)
            self.theSession.message("---------------------------")
    
            # creates and display error message dialog.
            anErrorMessage = "Couldnot change variablereference"
            anErrorTitle = "Error"
            anErrorWindow = ConfirmWindow(OK_MODE,anErrorMessage,anErrorTitle)
            self.update()
            
        
    def __cellEdited( self, *args ):
        aNewValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        column = args[3]
        if column == COEF_COLUMN:
            #check whether it is integer
            if not operator.isNumberType( aNewValue):
                anErrorWindow = ConfirmWindow(OK_MODE,"Coefficient should be numeric.","Error")
                return
        self.theListStore.set_value( anIter, column, aNewValue )
        self.__setValue()
        
    def __buttonPressed( self, *args ):

        event = args[1]
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
        else:
            realFullID = self.__getAbsoluteReference( selectedFullID )
            isFullIDReal = self.__doesExistEntity( realFullID )
            if isFullIDReal:
                return realFullID
            else:
                return None


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


    def __openNewAction( self, *args ):
        aFullIDString = args[1]

        theFullPNList =  [convertFullIDToFullPN( createFullID( aFullIDString ) )]  
        self.theSession.thePluginManager.createInstance( "PropertyWindow", 
                       theFullPNList )


    def __openAction ( self, *args ):
        aFullIDString = args[1]
        theFullPNList =  [convertFullIDToFullPN( createFullID( aFullIDString ) )]  
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
        aVarref = self.theListStore.get_value( anIter, FULLID_COLUMN )
        return  self.__getAbsoluteReference( aVarref )
        
    
    def __getSelectedIter( self ):
        anIter = self.theTreeView.get_selection().get_selected()[1]
        return anIter
        

        
    def __doesExistEntity( self, anEntity ):
        try:
            self.theSession.createEntityStub( anEntity ).getClassname()
        except:
            return False
        else:
            return True


    def __getAbsoluteReference( self, aVariableRef ):
        aVariable = aVariableRef.split(':')

        if self.__isAbsoluteReference( aVariableRef ):
            aVariable [0] = "Variable"
            return ":".join( aVariable)
        if aVariable[1][0] == '/':
            # absolute ref
            absolutePath = aVariable[1]
        elif aVariable[1][0] == '.':
            aProcess = self.theFullIDString.split(':')[1]
            aProcessPath = aProcess.split('/')
            while True:
                if len(aProcessPath) == 0:
                    break
                if aProcessPath[0] == '':
                    aProcessPath.__delitem__(0)
                else:
                    break
            aVariablePath = aVariable[1].split('/')
            absolutePath = ''
            while aVariablePath != []:
                pathString =  aVariablePath.pop()
                if pathString == '.':
                    break
                elif pathString == '..':
                    if len(aProcessPath) == 0:
                        return aVariableRef
                    aProcessPath.pop()
                else:
                    absolutePath =  pathString + '/' + absolutePath
            oldPath = '/' + '/'.join(aProcessPath)
            absolutePath = absolutePath.rstrip('/')
            if oldPath != '/' and absolutePath != '':
                oldPath +='/'
            absolutePath =  oldPath + absolutePath
    
        else:
            return aVariableRef
    
        return "Variable" + ':' + absolutePath + ':' + aVariable[2]
    
    
    
    def __isAbsoluteReference(self, aVariableRef ):
        aList = aVariableRef.split(':')
        return aList[1][0] == '/'
    
