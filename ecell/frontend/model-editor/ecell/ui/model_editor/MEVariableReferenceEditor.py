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
#
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


import os
import os.path

import gtk
import gobject

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.ViewComponent import *
from ecell.ui.model_editor.FullIDBrowserWindow import *

class MEVariableReferenceEditor(ViewComponent):
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):
    
        self.theParentWindow = aParentWindow
        # call superclass
        ViewComponent.__init__( self,   pointOfAttach, 'attachment_box', 'ListComponent.glade' )



        # set up liststore
        self.theListStore=gtk.ListStore( gobject.TYPE_STRING, gobject.TYPE_STRING, \
                    gobject.TYPE_STRING, gobject.TYPE_BOOLEAN, gobject.TYPE_STRING )

        self['theTreeView'].set_model(self.theListStore)

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__nameEdited )
        column=gtk.TreeViewColumn( 'Name', renderer, text = 0, editable = 3 )
        column.set_visible( True )
        self['theTreeView'].append_column(column)
        self.theNameColumn = column

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__valueEdited )
        column=gtk.TreeViewColumn( 'Variable', renderer, text = 1, editable = 3,foreground = 4 )
        column.set_visible( True )
        self['theTreeView'].append_column(column)

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__coefEdited )
        column=gtk.TreeViewColumn( 'Coefficient', renderer, text = 2, editable = 3 )
        column.set_visible( True )
        self['theTreeView'].append_column(column)
        self['theTreeView'].set_headers_visible(True)

        self.theListSelection =  self['theTreeView'].get_selection()
        self.theListSelection.set_mode( gtk.SELECTION_MULTIPLE )

        self.noUpdate = False
        # add signal handlers
        self.theFlags = [True, True, None, None, None, True]
        self.theSelection = []
        self.theProcessID = self.theParentWindow.theParentEntity
        self.theVarrefList = self.theParentWindow.thePropertyValue

        self.theSelectionTypeList = [ 'VarrefList' ]
        self.theModelEditor = self.theParentWindow.theModelEditor

        self['theTreeView'].connect('button-press-event' , self.__button_pressed )
        self['theTreeView'].connect('cursor-changed' , self.__cursor_changed )
        self.addHandlers ( { 'on_Add_clicked' : self.__add_clicked,\
                    'on_Delete_clicked' : self.__delete_clicked})
        self.update()

    def close( self ):
        # dereference Liststore and other member gtk objects
        self.theListSelection = None
        self.theListStore = None
        self.theNameColumn = None

        # call superclass close
        ListComponent.close( self )

    def getADCPFlags( self, aType ):

        self.theFlags[ ME_DELETE_FLAG ] = len( self.theSelection) > 0
        self.theFlags[ ME_BROWSE_FLAG ] = len( self.theSelection ) == 1
        return self.theFlags

    def getMenuItems( self ):
        aMenu = []
        aMenu.append( [ "add new", True ] )
        aMenu.append( [ "delete", len( self.theSelection ) > 0 ] )
        aMenu.append( ["browse", len( self.theSelection ) == 1 ] )
        if len( self.theSelection ) == 1:
            aVariable = self.getVarref(self.theSelection[0])[ME_VARREF_FULLID] 

            if isAbsoluteReference( aVariable ):
                aMenu.append( [ "turn relative", True ] )
            elif isRelativeReference( aVariable ):
                aMenu.append( [ "turn absolute", True ] )
        return aMenu

    def getVarref( self, aName ):
        for aVarref in self.theVarrefList:
            if aVarref[ME_VARREF_NAME] == aName:
                return aVarref
        return None

    def turn_absolute( self ):
        aVarref = self.getVarref(self.theSelection[0])
        try:
            aVarref[ME_VARREF_FULLID] = getAbsoluteReference( self.theProcessID, aVarref[ME_VARREF_FULLID] )
        except:
            self.theModelEditor.printMessage("Reference %s cannot be turned into absolute"%aVarref[ME_VARREF_FULLID] )
        self.update()

    def turn_relative( self ):
        aVarref = self.getVarref(self.theSelection[0])
        try:
            aVarref[ME_VARREF_FULLID] = getRelativeReference( self.theProcessID, aVarref[ME_VARREF_FULLID] )
        except:
            self.theModelEditor.printMessage("Reference %s cannot be turned into relative"%aVarref[ME_VARREF_FULLID] )
        self.update()

    def update( self ):
        """
        in:  None update without condition
        """

        self.__buildList()

        # restore selection
        self.restoreSelection()

    def changeSelection( self, aNameList, userSelect = False ):
        """
        in: varrefnamelist
        """
        # if cannot change select nothing
        self.theSelection = []
        for aVarref in self.theVarrefList:
            if aVarref[0] in aNameList:
                self.theSelection.append( aVarref[0] )  


        # change physically selected row if not user selected
        if not userSelect:
            aNameList = []
            for aSelection in self.theSelection:
                aNameList.append( aSelection )

            self.__selectRows( aNameList )

    def restoreSelection( self ):

        # call changeselection with stored selection
        self.changeSelection( self.theSelection )

    def selectByUser( self ):

        # get selected sysid
        aSelectionList = copyValue( self.__getSelection() )

        # call changeselection
        self.changeSelection( aSelectionList, True )

    def getValue( self):
        return copyValue( self.theVarrefList )

    def add_new ( self ):
        newName = self.getUniqueVarrefName()
        aFullIDBrowserWindow = FullIDBrowserWindow( self, convertSysPathToSysID( self.theProcessID.split(':')[1] ) )
        aVariableRef = aFullIDBrowserWindow.return_result()
        if aVariableRef == None:
            #aVariableRef = '.'
            return
        if getFullIDType( aVariableRef ) != ME_VARIABLE_TYPE:
            return
        
        if isAbsoluteReference( aVariableRef ):
            aVariableRef = getRelativeReference( self.theProcessID,  aVariableRef )
        
        #aVariableRef = aVariableRef.replace( ME_VARIABLE_TYPE, '', 1)
        aVarref = [ newName, aVariableRef, 0 ]
        self.theVarrefList.append( aVarref )
        self.update()
        
    def delete ( self ):
        aVarref = self.getVarref( self.theSelection[0] )
        self.theVarrefList.remove( aVarref )
        self.update()

    def browse( self ):
        aVarref = self.getVarref( self.theSelection[0] )
        aFullIDBrowserWindow = FullIDBrowserWindow( self, aVarref[ME_VARREF_FULLID] )
        result = aFullIDBrowserWindow.return_result()
        if result != None:
            if getFullIDType( result ) != ME_VARIABLE_TYPE:
                return
            if isAbsoluteReference( result ):
                result = getRelativeReference( self.theProcessID, result )
            aVarref[ME_VARREF_FULLID] = result
            self.update()

    def getUniqueVarrefName( self ):
        i = 0
        while True:
            newName = 'X' + str(i)
            if self.getVarref( newName) == None:
                return newName
            i += 1

    def getParentWindow( self ):
        return self.theParentWindow

    #########################################
    #    Private methods/Signal Handlers    #
    #########################################

    def __button_pressed( self, *args ):
        # when any button is pressed on list
        if args[1].button == 3:
            self.theModelEditor.createPopupMenu( self,  args[1] )
            return True

    def __cursor_changed( self, *args ):
        # when row is selected in list
        if self.noUpdate:
            return
        self.selectByUser()

    def __add_clicked( self, *args ):
        self.add_new()

    def __delete_clicked( self, *args ):
        self.delete()

    def __buildList( self ):
        """
        clear and build list
        """
        self.theListSelection.unselect_all()
        self.theListStore.clear()
        for aVarref in self.theVarrefList:
            anIter = self.theListStore.append()
            self.theListStore.set_value ( anIter, 0, aVarref[ME_VARREF_NAME] )
            self.theListStore.set_value ( anIter, 1, aVarref[ME_VARREF_FULLID] )
            self.theListStore.set_value ( anIter, 2, aVarref[ME_VARREF_COEF] )
            self.theListStore.set_value ( anIter, 3, True )
            aFullID = aVarref[ ME_VARREF_FULLID ]

            refValid = True
            if not isAbsoluteReference( aFullID ):
                try:
                    aFullID = getAbsoluteReference( self.theProcessID, aFullID )

                except:
                    refValid = False

            if refValid:
                refValid = self.theModelEditor.getModel().isEntityExist( aFullID )
            foreColor = "black"
            if not refValid:
                foreColor = "red"
            self.theListStore.set_value ( anIter, 4, foreColor )

    def __deleteRows( self, aNameList ):
        """
        in: list of string aNameList
        """

        pass

    def __getSelection( self ):
        """
        returns namelist
        """
        aPathList = self.__getSelectedRows()
        aNameList = []
        for aPath  in aPathList:
            anIter = self.theListStore.get_iter ( aPath )
            aNameList.append ( self.theListStore.get_value( anIter, 0 ) )
        return aNameList    

    def __selectRows( self, aNameList, forEdit = False ):
        """
        in: list of string aNameList
            bool forEdit can only go edit mode if only one name is in namelist
        """
        if len( aNameList ) == 0:
            self.theListSelection.unselect_all()
            return
        elif len( aNameList ) == 1:
            anIter = self.theListStore.get_iter_first()
            while anIter != None:
                aName = self.theListStore.get_value (anIter, 0)
                if aNameList.__contains__( aName ):
                    self.theListSelection.select_iter( anIter )
                    aPath = self.theListStore.get_path ( anIter )
                    self.noUpdate = True
                    self['theTreeView'].set_cursor( aPath, self.theNameColumn, forEdit )
                    self.noUpdate = False
                    return
                anIter = self.theListStore.iter_next( anIter )

        else:
            anIter = self.theListStore.get_iter_first()
            while anIter != None:
                aName = self.theListStore.get_value (anIter, 0)
                if aNameList.__contains__( aName ):
                    self.theListSelection.select_iter( anIter )

    def __getIter( self, aName ):
        """
        in: str aName
        """
        pass

    def __getSelectedRows( self ):
        """
        returns list of gtkPath
        """
        self.__thePathList = []
        self.theListSelection.selected_foreach( self.__foreachCallBack )
        return self.__thePathList

    def __foreachCallBack( self, *args ):
        """
        args[0] TreModel
        args[1] path
        args[2] iter
        """

        self.__thePathList.append( args[1] )

    def __nameEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """

        newName = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        oldName = self.theListStore.get_value( anIter, 0 )
        if oldName == newName:
            return
        if self.getVarref( newName )  == None:
            oldVarref = self.getVarref( oldName )
            oldVarref[ME_VARREF_NAME] = newName
            self.update()
        else:
            self.theModelEditor.printMessage("Cannot rename to %s because it is already used."%newName, ME_ERROR )

    def __valueEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """

        newValue = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        oldName = self.theListStore.get_value( anIter, 0 )
        oldVarref = self.getVarref( oldName )
        newTupple = newValue.split(':')
        valid = False
        if len(newTupple) == 3:
            if newTupple[1][0] in ['.','/']:
                valid = True
        if valid:
            oldVarref[ME_VARREF_FULLID] = newValue
            self.update()
        else:
            self.theModelEditor.printMessage("%s is not a FullID."%newValue, ME_ERROR )

    def __coefEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """

        newCoef = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        oldName = self.theListStore.get_value( anIter, 0 )
        oldVarref = self.getVarref( oldName )
        try:
            newCoef = int(newCoef)
        except:
            self.theModelEditor.printMessage("Coefficient should be of integer value.", ME_ERROR )
        oldVarref[ME_VARREF_COEF] = newCoef
        self.update()
