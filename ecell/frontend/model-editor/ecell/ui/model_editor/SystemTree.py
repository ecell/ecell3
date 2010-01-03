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
from ecell.ui.model_editor.EntityCommand import *
from ecell.ui.model_editor.AutoLayout import *

class SystemTree(ViewComponent):
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):

        # call superclass
        ViewComponent.__init__( self, pointOfAttach,
         'attachment_box', 'ListComponent.glade' )
        self.theParentWindow = aParentWindow

        # set up systemtree
        self.theSysTreeStore=gtk.TreeStore( gobject.TYPE_STRING, gobject.TYPE_BOOLEAN )
        self['theTreeView'].set_model(self.theSysTreeStore)
        renderer = gtk.CellRendererText()

        renderer.connect('edited', self.__cellEdited)
        column=gtk.TreeViewColumn( 'System Tree',
                       renderer,
                       text=0, editable = 1 )
        column.set_visible( True )
        self['theTreeView'].append_column(column)
        self.theColumn = column
        self.theSysSelection =  self['theTreeView'].get_selection()
        self.theSysSelection.set_mode( gtk.SELECTION_MULTIPLE )
        self['theTreeView'].set_headers_visible( False )
        self.theModelEditor = self.theParentWindow.theModelEditor
        # set up variables
        self.theSelection = []
        self.theSelectionTypeList = [ME_SYSTEM_TYPE, ME_VARIABLE_TYPE, ME_PROCESS_TYPE, ME_PROPERTY_TYPE ]

        # build up tree recursively
        self.__buildTree ( None, MS_SYSTEM_ROOT )

        # determine flags
        self.theFlags = [ True, True, True, True ]
        self.userSelect = True
        self.noActivate = False
        self['theTreeView'].connect('button-press-event' , self.__button_pressed )
        self['theTreeView'].connect('cursor-changed' , self.__cursor_changed )
        self.addHandlers ( { 'on_Add_clicked' : self.__add_clicked,\
                    'on_Delete_clicked' : self.__delete_clicked})

    def getMenuItems(self):        
        aMenu = ViewComponent.getMenuItems(self)
        aFlags = self.getADCPFlags(self.theSelectionTypeList)
         

        fullPNList = map( lambda x:x+':', self.getSelectedIDs() )
        
        aMenu.append(["generateLayout",  len(fullPNList) == 1])

        tracerMenu = self.theModelEditor.theRuntimeObject.createTracerSubmenu( fullPNList )
        aMenu.append([None, tracerMenu ])

        return aMenu


    def generateLayout(self):
        #print self.theSelection

        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return
        layoutName = self.theModelEditor.theLayoutManager.getUniqueLayoutName()
        #print layoutName
        
        self.theModelEditor.theMainWindow.displayHourglass()
        self.theAutoLayout = AutoLayout(self.theModelEditor,layoutName,self.theSelection)
        self.theModelEditor.theMainWindow.resetCursor()


    def getPasteableTypes( self ):
        return self.theSelectionTypeList


    def getParentWindow( self ):
        return self.theParentWindow
    

    def close( self ):
        # dereference Liststore and other member gtk objects
        self.theSysSelection = None
        self.theSysTreeStore = None
        self.theColumn = None

        # call superclass close
        ViewComponent.close( self )



    def getSelectedIDs( self ):
        #print 'System ::/"                
        return copyValue( self.theSelection )



    def update ( self, aSystemFullID ):
        """
        in: string aSystemFullID
        """

        if aSystemFullID == None:
            aSystemFullID = MS_SYSTEM_ROOT
        # delete TreeStore down from aSystemPath
        self.noActivate = True
        self.__removeTree( aSystemFullID )
        self.noActivate = False
        if aSystemFullID == MS_SYSTEM_ROOT:
            anIter = None
        else:
            anIter = self.__getIter( getParentSystemOfFullID( aSystemFullID ) )


        # recursively build TreeStore down from aSystemFullID
        self.__buildTree ( anIter, aSystemFullID )
    
        # restore selection
        self.restoreSelection()


    def changeSelection( self, aSysIDList, userSelect = False ):
        """
        in: aSysID
            userSelect- True if user selected the row
        """
        # change 
       
        self.theSelection = aSysIDList

        if not userSelect:
            self.__unselectRows()

        removeList = []
        for aSysID in self.theSelection:
            # if cannot change select root
            anIter = self.__getIter(  aSysID  )

            if anIter == None:
                removeList.append( aSysID )
                continue
    
            # change physically selected row if not user selected
            if not userSelect:
                self.__selectRow( anIter )
        for aSysID in removeList:
            self.theSelection.remove(aSysID )

        if len(self.theSelection) == 0:
            self.theSelection = [ MS_SYSTEM_ROOT ]
            anIter = self.__getIter( self.theSelection[0] )

            if not userSelect:
                self.__selectRow( anIter )



    def getADCPFlags( self, aType ):
        self.theFlags[ ME_PASTE_FLAG ] = aType in self.theSelectionTypeList        
        if len(self.theSelection) == 0:
            self.theFlags[ME_DELETE_FLAG] = False
        elif len(self.theSelection) == 1:
            if self.theSelection[0] == MS_SYSTEM_ROOT:
                self.theFlags[ME_DELETE_FLAG] = False
                self.theFlags[ME_COPY_FLAG ] = False
            else:
                self.theFlags[ME_DELETE_FLAG] = True
                self.theFlags[ME_COPY_FLAG ] = True
        else:
            self.theFlags[ME_DELETE_FLAG] = True
        return self.theFlags


    def selectByUser( self ):
        # get selected sysid
        if not self.userSelect:
            return

        self.theSelection = self.__getSelection()

        # update parentwindow entitylist
        self.theParentWindow.update()



    def restoreSelection( self ):

        # call changeselection with stored selection
        self.changeSelection( self.theSelection )


    def copy( self ):
        self.selectByUser()

        # create command
        aCommand = CopyEntityList( self.theParentWindow.theModelEditor, self.theSelection )

        # execute
        self.theParentWindow.theModelEditor.doCommandList( [ aCommand ] )


    def cut ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        self.selectByUser()

        # create command
        aCommand = CutEntityList( self.theParentWindow.theModelEditor, self.theSelection )

        # execute
        self.theParentWindow.theModelEditor.doCommandList( [ aCommand ] )
        

    def paste ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        self.selectByUser()
        aCommandList = []
        aBuffer = self.theModelEditor.getCopyBuffer()

        if aBuffer.getType() == ME_PROPERTY_TYPE:
            for aSelection in self.theSelection:
                aCommandList.append( PasteEntityPropertyList( self.theModelEditor, aSelection, aBuffer ) )

        else:
            for aSelection in self.theSelection:
                pastePath = convertSysIDToSysPath( aSelection )
                aCommandList.append( PasteEntityList( self.theModelEditor, aBuffer, pastePath ) )
        self.theModelEditor.doCommandList( aCommandList )


    def add_new ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        self.selectByUser()

        # get unique name from modeleditor
        # change selection
        aPath = convertSysIDToSysPath ( self.theSelection[0] )
        newName = self.theModelEditor.getUniqueEntityName( ME_SYSTEM_TYPE, aPath )
        newID = ME_SYSTEM_TYPE + ':' + aPath + ':' + newName

        # call addnew in modeleditor
        aCommand = CreateEntity( self.theModelEditor, newID, DM_SYSTEM_CLASS )
        self.theSelection = [ newID ]
        self.__unselectRows()
        self.theModelEditor.doCommandList ( [ aCommand ] )
        ############################## Add cheCk here ###############################3
        # open for edit

        anIter = self.__getIter ( self.theSelection[0] )
        self.noActivate = True
        self.__selectRow( anIter, True )
        self.noActivate = False

    
    def delete ( self ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        self.selectByUser()

        # root cannot be selected
        deleteList = []
        for aSysID in self.theSelection:
            if aSysID != MS_SYSTEM_ROOT:
                deleteList.append( aSysID )

        aCommand = DeleteEntityList( self.theModelEditor, deleteList )
        
        self.theSelection = [ MS_SYSTEM_ROOT ]
        self.theModelEditor.doCommandList ( [ aCommand ] )
        

    def rename ( self, newName, anIter ):
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        # if nothing changed make nothing
        #newSelection = self.__getSelection()
        #oldID = newName
        #for anOldID in self.theSelection:
        #   if anOldID not in newSelection:
        #       oldID = anOldID
        #       break

        oldName = self.theSysTreeStore.get_value( anIter, 0 )
        if oldName == newName:
            return
        if not isIDEligible( newName ):
            self.theModelEditor.printMessage( "Only alphanumeric characters and _ are allowed in system ids", ME_ERROR )
            self.theSysTreeStore.set_value( anIter, 0, oldName )
            return
        #oldTuple = [ ME_SYSTEM_TYPE, convertSysIDToSysPath( self.theDisplayedSysID ), oldName ]
        #oldID = ':'.join( oldTuple )
        oldID = self.__getSysID( anIter )
    
        newTuple = oldID.split(':')
        newTuple[2] = newName
        newID = ':'.join( newTuple )

        aCommand = RenameEntity( self.theModelEditor, oldID, newID )
        if aCommand.isExecutable():
            self.theSelection = [ newID ]
            self.theModelEditor.doCommandList( [ aCommand ] )
        else:
            self.theSysTreeStore.set_value( anIter, 0, oldName )



    #########################################
    #    Private methods/Signal Handlers    #
    #########################################

    def __button_pressed( self, *args ):
        # when any button is pressed on list
        self.theModelEditor.setLastUsedComponent( self )
        self.theParentWindow.setLastActiveComponent( self )
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
        self.theSysSelection.unselect_all()


    def __buildTree( self, fromIter, fromSysID ):
        """
        in: gtk.TreePath fromIter, SysID, fromSysID     
        builds up the systemstore down from SysID, down from the fromIter
        if fromIter None, creates it from the top
        """
        # creates an entry for SysID


        fromTuple = fromSysID.split(':')
        anIter = self.theSysTreeStore.append( fromIter )
        self.theSysTreeStore.set_value( anIter, 0, fromTuple[2] )
        editable = True
        if fromTuple[2] == '/':
            editable = False
        self.theSysTreeStore.set_value( anIter, 1, editable )       

        # gets system entities of SysID
        parentSysPath = convertSysIDToSysPath ( fromSysID  )
        subSystemList = self.theParentWindow.theModelEditor.getModel().\
            getEntityList ( ME_SYSTEM_TYPE,  parentSysPath )
        subSystemList = list( subSystemList )
        # call itself recursively
        subSystemList.sort( lambda x,y:(x>y)-(x<y) )
        for aSubSystem in subSystemList:
            self.__buildTree( anIter, ":".join( [ME_SYSTEM_TYPE, parentSysPath, aSubSystem] ) )

        
    def __removeTree( self, fromSysID ):
        """
        in: string fromSysID
        """
        anIter = self.__getIter( fromSysID )
        if anIter != None:
            self.theSysTreeStore.remove( anIter )

    
    def __getSelection( self ):
        """
        returns selected sysID
        """
        aPathList = self.__getSelectedRows()
        return_list = []

        for aPath in aPathList:
            anIter = self.theSysTreeStore.get_iter( aPath )            
            return_list.append( self.__getSysID( anIter ) )
       
        return return_list


    def __getSelectedRows( self ):
        """
        returns list of gtkPath
        """
        self.__thePathList = []
        self.theSysSelection.selected_foreach( self.__foreachCallBack )
        return self.__thePathList


    def __foreachCallBack( self, *args ):
        """
        args[0] TreModel
        args[1] path
        args[2] iter
        """

        self.__thePathList.append( args[1] )


    def __getIter( self, aSysID ):
        aSysPath = convertSysIDToSysPath( aSysID )
        return self.__getIter2( aSysPath )


    def __getIter2( self, aSysPath, anIter = None ):
        """
        returns iter of string aSysPath or None if not available
        """

        if anIter == None:
            anIter = self.theSysTreeStore.get_iter_first()
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
        numChildren = self.theSysTreeStore.iter_n_children( anIter )
        isFound = False
        for i in range( 0, numChildren):
            childIter = self.theSysTreeStore.iter_nth_child( anIter, i )

            if self.theSysTreeStore.get_value( childIter, 0) == firstTag:
                isFound = True

                break

        # if not found return None
        if not isFound:

            return None

        # if remainder is '' return iter
        if aRemainder == '':

            return childIter




        # return recursive remainder with iter
        return self.__getIter2( aRemainder, childIter )
        


    def __getSysID( self, anIter ):
        """
        returns SysID belonging to an iter
        """
        anID = self.theSysTreeStore.get_value( anIter, 0 )

        aSystemPath = ''
        while True:
            parentIter = self.theSysTreeStore.iter_parent( anIter )
            if parentIter == None:
                return ':'.join( [ ME_SYSTEM_TYPE, aSystemPath, anID ] )
            newPath = str(self.theSysTreeStore.get_value( parentIter, 0 ))
            if newPath != '/':
                newPath = '/' + newPath
                aSystemPath = newPath + aSystemPath
            elif aSystemPath[0:1] != '/':
                aSystemPath = '/' + aSystemPath
            anIter = parentIter



    def __selectRow ( self, anIter, forEdit = False ):
        """
        in:     gtkIter anIter
            bool forEdit
        """
        aPath = self.theSysTreeStore.get_path( anIter )
        self.__expandRow( aPath )
        self.theSysSelection.select_iter( anIter )
        if forEdit == True:
            self.noActivate = True
            self['theTreeView'].set_cursor( aPath, self.theColumn, forEdit )
        self.noActivate = False
            

    def __expandRow( self, aPath ):
        """
        in: gtktreePath aPath
        """
        if not self['theTreeView'].row_expanded( aPath ):

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
            self['theTreeView'].expand_row( aPath, False )


    
    def __cellEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """
        newName = args[2]
        anIter = self.theSysTreeStore.get_iter_from_string( args[1] )
        self.rename ( newName, anIter )
