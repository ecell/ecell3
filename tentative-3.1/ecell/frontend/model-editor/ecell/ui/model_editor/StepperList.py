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
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.ViewComponent import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.StepperCommand import *

class StepperList(ViewComponent):
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):
        self.theParentWindow = aParentWindow
        # call superclass
        ViewComponent.__init__( self,  pointOfAttach,\
         'attachment_box', 'ListComponent.glade' )
        # set up liststore
        self.theListStore=gtk.ListStore( gobject.TYPE_STRING, gobject.TYPE_BOOLEAN )
        self['theTreeView'].set_model(self.theListStore)
        renderer = gtk.CellRendererText()

        renderer.connect('edited', self.__cellEdited)
        column=gtk.TreeViewColumn( 'System Tree',
                       renderer,
                       text=0, editable = 1 )
        column.set_visible( True )
        self['theTreeView'].append_column(column)
        self.theColumn = column
        self.theListSelection =  self['theTreeView'].get_selection()
        self.theListSelection.set_mode( gtk.SELECTION_MULTIPLE )
        self['theTreeView'].set_headers_visible( False )
        self.theModelEditor = self.theParentWindow.theModelEditor
       
        # set up variables
        self.userSelect = True
        self.noActivate = False
        self.theFlags = [ True, True, True, True ]
        self.theSelection = []
        self.theSelectionTypeList = [ ME_STEPPER_TYPE, ME_PROPERTY_TYPE ]
        self['theTreeView'].connect('button-press-event' , self.__button_pressed )
        self['theTreeView'].connect('cursor-changed' , self.__cursor_changed )
        self.addHandlers ( { 'on_Add_clicked' : self.__add_clicked,\
                    'on_Delete_clicked' : self.__delete_clicked})


    def getPasteableTypes( self ):
        return self.theSelectionTypeList


    def getParentWindow( self ):
        return self.theParentWindow


    def close( self ):
        # dereference Liststore and other member gtk objects
        self.theListSelection = None
        self.theListStore = None
        self.theColumn = None

        # call superclass close
        ViewComponent.close( self )



    def update( self ):
        """
        just destroy and rebuild, reselet selection
        """

        # if changed system displayed clear liststore and rebuild list
        self.__buildList()


        # restore selection
        self.restoreSelection()



    def getSelectedIDs( self ):
        """
        returns list of selected IDs
        """
        return copyValue( self.theSelection )



    def changeSelection( self, aStepperIDList, userSelect = False ):
        """
        in:  stepperIDList aStepperIDList
        """
        # change self.theSelection
        # if cannot change select nothing
        self.theSelection = []
        for aStepperID in aStepperIDList:
            if aStepperID in self.theModelEditor.getModel().getStepperList():
                self.theSelection.append( aStepperID )
        # modify flags

        # change physically selected row if not user selected
        if not userSelect:
            self.__selectRows( self.theSelection )




    def getADCPFlags( self, aType ):
        self.theFlags[ ME_COPY_FLAG ] = len( self.theSelection) > 0
        self.theFlags[ ME_DELETE_FLAG ] = len( self.theSelection) > 0
        self.theFlags[ ME_PASTE_FLAG ] = aType in self.theSelectionTypeList
        if aType == ME_PROPERTY_TYPE:
            self.theFlags[ ME_PASTE_FLAG ] = len( self.theSelection) > 0
        return self.theFlags


    def restoreSelection( self ):

        # call changeselection with stored selection
        self.changeSelection( self.theSelection )


    def selectByUser( self ):

        # get selected sysid
        self.theSelection = copyValue( self.__getSelection() )       
        # update parentwindow propertylist
        self.theParentWindow.update()

        # call changeselection
        #self.changeSelection( aSelectionList, True )


    def copy( self ):

        # create command
        aCommand = CopyStepperList( self.theModelEditor, self.theSelection )

        # execute
        self.theModelEditor.doCommandList( [ aCommand ] )


    def cut ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        # create command
        aCommand = CutStepperList( self.theParentWindow.theModelEditor, self.theSelection )

        # execute
        self.theParentWindow.theModelEditor.doCommandList( [ aCommand ] )


    def paste ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        aCommandList = []
        aBuffer = self.theModelEditor.getCopyBuffer()
        if aBuffer.getType() == ME_PROPERTY_TYPE:
            for aSelection in self.theSelection:
                aCommandList.append( PasteStepperPropertyList( self.theModelEditor, aSelection, aBuffer ) )

        else:
            aCommandList.append( PasteStepperList( self.theModelEditor, aBuffer ) )
            self.theSelection = aBuffer.getStepperList()
        self.theModelEditor.doCommandList( aCommandList )


    def add_new ( self ):

        # get unique name from modeleditor
        # change selection
        # call addnew in modeleditor
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        newClass = self.theModelEditor.getDefaultStepperClass()

        newName = self.theModelEditor.getUniqueStepperName ()

        aCommand = CreateStepper( self.theModelEditor,newClass, newName )
        self.theSelection = [ newName ]
        self.__unselectRows()
        self.theModelEditor.doCommandList ( [ aCommand ] )
    
        # open for edit
        self.noActivate = True
        self.__selectRows( [ newName ], True )
        self.noActivate = False

    def delete ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return
        aCommand = DeleteStepperList( self.theModelEditor, self.theSelection )
        
        self.theSelection = []
        self.theModelEditor.doCommandList ( [ aCommand ] )


    def rename ( self, newName, anIter ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        # if nothing changed make nothing
        oldName = self.theListStore.get_value( anIter, 0 )

        if oldName == newName:
            return
        if not isIDEligible( newName ):
            self.theModelEditor.printMessage( "Only alphanumeric characters and _ are allowed in stepper ids!", ME_ERROR )
            self.theListStore.set_value( anIter, 0, oldName )
            return
        aCommand = RenameStepper( self.theModelEditor, oldName, newName )
        if aCommand.isExecutable():
            self.theSelection = [ newName ]
            self.theModelEditor.doCommandList( [ aCommand ] )
        else:
            self.theListStore.set_value( anIter, 0, oldName )





    #########################################
    #    Private methods/Signal Handlers    #
    #########################################


    def __button_pressed( self, *args ):
        # when any button is pressed on list
        self.theModelEditor.setLastUsedComponent( self )
        if args[1].button == 3:
            self.theModelEditor.createPopupMenu( self, args[1] )
            return True


    def __cursor_changed( self, *args ):
        # when row is selected in list

        if self.noActivate:
            return
        self.theModelEditor.setLastUsedComponent( self )
        self.theParentWindow.setLastActiveComponent( self )
        self.selectByUser()



    def __add_clicked( self, *args ):
        self.add_new()
        

    def __delete_clicked( self, *args ):
        self.delete()
        

    def __unselectRows( self ):

         self.theListSelection.unselect_all()
        

        
    def __buildList( self ):
        """
        clear and build list
        """
        self.noActivate = True
        self.theListSelection.unselect_all()
        self.theListStore.clear()
        anIDList = self.theModelEditor.getModel().getStepperList ( )
        
        self.__addRows( anIDList )
        self.noActivate = False
        
    
    def __addRows( self, aNameList ):
        """
        in: list of string aNameList
        """
        for aName in aNameList:
            anIter = self.theListStore.append(  )
            self.theListStore.set_value ( anIter, 0 , aName )
            self.theListStore.set_value ( anIter, 1 , True )
        



    def __deleteRows( self, aNameList ):
        """
        in: list of string aNameList
        """

        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            aName = self.theListStore.get_value( anIter, 0 )
            if aNameList.__contains__( aName ):
                self.theListStore.remove( anIter )
            anIter = self.theListStore.iter_next( anIter )


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

        

    def __selectRows( self, aNameList, forEdit = False, doSelect = False ):
        """
        in: list of string aNameList
            bool forEdit can only go edit mode if only one name is in namelist
            bool doSelect forces select row
        """
        if len( aNameList ) == 0:
            return
        elif len( aNameList ) == 1:
            anIter = self.theListStore.get_iter_first()
            while anIter != None:
                aName = self.theListStore.get_value (anIter, 0)
                if aNameList.__contains__( aName ):
                    self.theListSelection.select_iter( anIter )

                    if forEdit == False and not doSelect:
                        self.noActivate = True
                    aPath = self.theListStore.get_path ( anIter )
                    self['theTreeView'].set_cursor( aPath, self.theColumn, forEdit )
                    self.noActivate = False
                    return
                anIter = self.theListStore.iter_next( anIter )

        else:
            anIter = self.theListStore.get_iter_first()
            while anIter != None:
                aName = self.theListStore.get_value (anIter, 0)
                if aNameList.__contains__( aName ):
                    self.theListSelection.select_iter( anIter )
                anIter = self.theListStore.iter_next(anIter)


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


    def __cellEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """

        newName = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        self.rename ( newName, anIter )
        
