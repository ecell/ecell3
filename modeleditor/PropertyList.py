#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of E-CELL Model Editor package
#
#               Copyright (C) 1996-2003 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPvalueOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from Utils import *
import gtk
import gobject

import os
import os.path
import operator

from PropertyEditor import *
from ModelEditor import *
from ViewComponent import *
from Constants import *
from EntityCommand import *
from StepperCommand import *

PROP_SAVED = "grey0"
PROP_CAN_BE_SAVED = "grey35"
PROP_WILL_NOT_SAVED = "grey70"
CHGD_PROP_IND_TYPE = gobject.TYPE_STRING

class PropertyList( ViewComponent ):

    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):

        self.theParentWindow = aParentWindow
        # call superclass
        ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'ListComponent.glade' )

        # set up liststore
        self.theListStore=gtk.ListStore( gobject.TYPE_STRING, gobject.TYPE_STRING, \
                    gobject.TYPE_BOOLEAN, gobject.TYPE_BOOLEAN, \
                    gobject.TYPE_BOOLEAN, gobject.TYPE_BOOLEAN,\
                    gobject.TYPE_BOOLEAN, gobject.TYPE_BOOLEAN,\
                     gobject.TYPE_STRING,CHGD_PROP_IND_TYPE)
        self.theListStore.set_sort_column_id(9, gtk.SORT_ASCENDING)
        self['theTreeView'].set_model(self.theListStore)
        sourceTargets = [ (DND_PROPERTYLIST_TYPE, 0, 800)]
        self['theTreeView'].enable_model_drag_source( gtk.gdk.BUTTON1_MASK, sourceTargets, gtk.gdk.ACTION_COPY )
        self['theTreeView'].connect("drag-data-get", self.__drag_data_get )
        self.noActivate = False
        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__nameEdited)
        column = gtk.TreeViewColumn( 'Name', renderer, text = 0, editable = 4, foreground = 9)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)
        self.theNameColumn = column

        renderer = gtk.CellRendererText()
        renderer.connect('edited', self.__valueEdited)
        column = gtk.TreeViewColumn( 'Value', renderer, text = 1, editable = 5, foreground = 9)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)
        self.theValueColumn = column

        renderer = gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( 'Settable', renderer, active = 2)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)


        renderer = gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( 'Deleteable', renderer, active = 3)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)

        renderer = gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( 'Loadable', renderer, active = 6)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)

        renderer = gtk.CellRendererToggle()
        column=gtk.TreeViewColumn( 'Saveable', renderer, active = 7)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)

        # add variable type column
        renderer = gtk.CellRendererText()
        column = gtk.TreeViewColumn( 'Type', renderer, text = 8, editable = 5, foreground = 9)
        column.set_visible( gtk.TRUE )
        column.set_resizable( gtk.TRUE )
        self['theTreeView'].append_column(column)
        self.theTypeColumn = column

        self['theTreeView'].set_headers_visible(gtk.TRUE)

        self.theListSelection =  self['theTreeView'].get_selection()
        self.theListSelection.set_mode( gtk.SELECTION_MULTIPLE )
        self.theListSelection.connect('changed',self.__setDeleteButton)

        # set up variables

        self.theFlags = [ True, True, True, True, True, False ]
        self.theSelection = []
        self.theDisplayedEntity = None
        self.theType = None
        self.theSelectionTypeList = [ ME_PROPERTY_TYPE ]
        self.theModelEditor = self.theParentWindow.theModelEditor

        self['theTreeView'].connect('button-press-event' , self.__button_pressed )
        self['theTreeView'].connect('cursor-changed' , self.__cursor_changed )

        self.addHandlers ( { 'on_Add_clicked' : self.__add_clicked,\
                    'on_Delete_clicked' : self.__delete_clicked})

        self.addButton=self.getWidget('Add')
        self.delButton=self.getWidget('Delete')
        self.update()             
        

    def __drag_data_get( self, *args ):
    	widget = args[0]
    	data_sel = args[2]
        filename = self.theModelEditor.theModelFileName 
        resultfilename = filename + '.dat'
        stringTupple = []
        for aName in self.theSelection:
            value = self.getPropertyValue( aName )
            fullpn = self.theDisplayedEntity + ":" + aName
            stringTupple.append( " ".join( [ filename, resultfilename, fullpn, str( value ), "0", "0" ] ) )
        selectionValue = ",".join( stringTupple )

    	data_sel.set( "text", 8, selectionValue )
    	
        
        
    def hideButtons( self ):
        self['attachment_box'].remove( self['hbox1'] )
        
    def close( self ):
        # dereference Liststore and other member gtk objects
        self.theListSelection = None
        self.theListStore = None
        self.theNameColumn = None
        self.theValueColumn = None
    
        # call superclass close
        ViewComponent.close( self )



    def getParentWindow( self ):
        return self.theParentWindow


    def getDisplayedEntity( self ):
        return self.theDisplayedEntity


    def getPropertyValue( self, aName ):
        try:
            if self.theType == 'Stepper':
                if self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[MS_GETTABLE_FLAG]:
                    return self.theModelEditor.getModel().getStepperProperty( self.theDisplayedEntity, aName )
                else:
                    return ''
            elif self.theType == 'Entity':
                fpn = self.theDisplayedEntity + ':' + aName
                if self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[MS_GETTABLE_FLAG]:

                    return self.theModelEditor.getModel().getEntityProperty( fpn )
                else:
                    return ''
            else:
                return None
        except:
            return None



    def getPropertyType ( self, aName ):
        #FIXME!
        # this whole method should be rewritten using DMINFO!
        #if self.theModelEditor.getMode() == ME_RUN_MODE:
        #    return DM_PROPERTY_STRING
        if self.theType == 'Stepper':
            aClass = self.theModelEditor.getModel().getStepperClassName( self.theDisplayedEntity )
#            return self.theModelEditor.getModel().getStepperPropertyType( self.theDisplayedEntity, aName )
        elif self.theType == 'Entity':
#            fpn = self.theDisplayedEntity + ':' + aName
            aClass = self.theModelEditor.getModel().getEntityClassName( self.theDisplayedEntity )
#            return self.theModelEditor.getModel().getEntityPropertyType( fpn )
        else:
            return None
        return self.theModelEditor.theDMInfo.getClassPropertyInfo ( aClass, aName, DM_PROPERTY_TYPE )

    def getPropertyChanged ( self, aName ):
        # if it is running all properties may change
        if self.theModelEditor.getMode() == ME_RUN_MODE:
            return 1

        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[ME_CHANGED_FLAG]
        elif self.theType == 'Entity':
            fpn = self.theDisplayedEntity + ':' + aName
            return self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[ME_CHANGED_FLAG]
        else:
            return None


    def getPropertySettable( self, aName ):
        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[ME_SETTABLE_FLAG]
        elif self.theType == 'Entity':
            fpn = self.theDisplayedEntity + ':' + aName
            return self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[ME_SETTABLE_FLAG]
        else:
            return None



    def getPropertyDeleteable( self, aName ):
        if self.theModelEditor.getMode() == ME_RUN_MODE:
            return False
        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[ME_DELETEABLE_FLAG]
        elif self.theType == 'Entity':
            fpn = self.theDisplayedEntity + ':' + aName
            return self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[ME_DELETEABLE_FLAG]
        else:
            return None

    
    def getPropertySaveable( self, aName ):
        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[ME_SAVEABLE_FLAG]
        elif self.theType == 'Entity':
            fpn = self.theDisplayedEntity + ':' + aName
            return self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[ME_SAVEABLE_FLAG]
        else:
            return None

    
    def getPropertyLoadable( self, aName ):
        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyAttributes( self.theDisplayedEntity, aName )[ME_LOADABLE_FLAG]
        elif self.theType == 'Entity':
            fpn = self.theDisplayedEntity + ':' + aName
            return self.theModelEditor.getModel().getEntityPropertyAttributes( fpn )[ME_LOADABLE_FLAG]
        else:
            return None



    def getPropertyList( self ):
        if self.theType == 'Stepper':
            return self.theModelEditor.getModel().getStepperPropertyList( self.theDisplayedEntity )
        elif self.theType == 'Entity':
           
            return self.theModelEditor.getModel().getEntityPropertyList( self.theDisplayedEntity  )
        else:
            return []

    def canAddNewProperty( self ):
        if self.theModelEditor.getMode() == ME_RUN_MODE:
            return False

        if self.theType == 'Stepper':
            aClass = self.theModelEditor.getModel().getStepperClassName( self.theDisplayedEntity )
        elif self.theType == 'Entity':
            aClass = self.theModelEditor.getModel().getEntityClassName( self.theDisplayedEntity )
        else:
            return False
        classInfoList = self.theModelEditor.getDMInfo().getClassInfoList( aClass )
        if DM_ACCEPTNEWPROPERTY in classInfoList:
            return self.theModelEditor.getDMInfo().getClassInfo( aClass,  DM_ACCEPTNEWPROPERTY )
        else:
            return False


    def update( self, mustRebuild = False ):
        """
        in:  None update without condition
        """
        # set button's sensitivity
        if not mustRebuild and self.theModelEditor.getMode() == ME_RUN_MODE:
            self.__updateList()
        if self.canAddNewProperty():
            self.addButton.set_sensitive(gtk.TRUE)
        else:
            self.addButton.set_sensitive(gtk.FALSE)
        self.delButton.set_sensitive(gtk.FALSE)

        
        self.__buildList()

        # restore selection

        self.restoreSelection()



    def getADCPFlags( self, aType ):
        self.theFlags[ ME_ADD_FLAG ] = self.canAddNewProperty()
        flag = False
        if  len( self.theSelection) > 0:
            for aName in self.theSelection:
                if self.getPropertyDeleteable( aName):
                    flag = True
                    break
        self.theFlags[ ME_DELETE_FLAG ] = flag

        self.theFlags[ ME_COPY_FLAG ] = len( self.theSelection) > 0

        self.theFlags[ ME_PASTE_FLAG ] = aType in self.theSelectionTypeList

        return self.theFlags


    def getMenuItems( self ):
        aMenu = ViewComponent.getMenuItems(self)
        editFlag = len( self.theSelection ) == 1
        if editFlag :
            editFlag = self.getPropertySettable( self.theSelection[0] )
        aMenu.append( [ "edit", editFlag ]  )
        if self.theType != "Stepper":
            tracerFpnList = []
            for fpn in self.getFullPNList():
                if self.getPropertyType( fpn.split(":")[3] ) in [ DM_PROPERTY_INTEGER, DM_PROPERTY_FLOAT ]:
                    tracerFpnList.append( fpn )
            if len( tracerFpnList ) > 0:
                tracerMenu = self.theModelEditor.theRuntimeObject.createTracerSubmenu( self.getFullPNList() )
                aMenu.append( [ None, tracerMenu ] )
        return aMenu



    def setDisplayedEntity ( self, aType, selectedID ):
        self.theDisplayedEntity = selectedID 
        self.theType = aType
        if self.theDisplayedEntity == None:
            self.theFlags = [False, False, False, False, False]
            self.theType = None
        else:
            self.theFlags = [True, True, True, True, True]

        #self.theSelection = []
        self.update( True )

    

    def getSelectedIDs( self ):
        """
        returns list of selected IDs
        """
        return copyValue( self.theSelection )



    def changeSelection( self, aNameList, userSelect = False ):
        """
        in: aNameList 
        """
        # change self.theSelection
        # if cannot change select nothing
        self.theSelection = []
        wholePropertyList = self.getPropertyList()
               
        for aPropertyID in wholePropertyList:
            if aPropertyID in aNameList: 
                           
                self.theSelection.append( aPropertyID )
                 
        # change physically selected row if not user selected
        if not userSelect:

            self.__selectRows( self.theSelection )
            
             
    def restoreSelection( self ):
        # call changeselection with stored selection

        self.changeSelection( self.theSelection )

    

    def selectByUser( self ):        
        # call changeselection
        #change to PropertyList tab in EntityWindow      
        self.changeSelection( copyValue( self.__getSelection() ), True ) 
       
                        
    def getFullPNList( self ):
        aFullPNList = []
        for aName in self.theSelection:
            aFullPNList.append(self.theDisplayedEntity + ':' + aName )     
        return aFullPNList
       
    

    def copy( self ):
        if self.theType == 'Stepper':
            aCommand = CopyStepperPropertyList( self.theModelEditor, 
self.theDisplayedEntity, self.theSelection )

        else:
            aCommand = CopyEntityPropertyList ( self.theModelEditor, self.getFullPNList() )

        self.theModelEditor.doCommandList( [ aCommand ] )


    def cut ( self ):
        if self.theType == 'Stepper':
            aCommand = CutStepperPropertyList( self.theModelEditor, 
self.theDisplayedEntity, self.theSelection )
        else:
            aCommand = CutEntityPropertyList ( self.theModelEditor, self.getFullPNList() )
        self.theModelEditor.doCommandList( [ aCommand ] )



    def paste ( self ):
        if self.theType == 'Stepper':
            aCommand = PasteStepperPropertyList( self.theModelEditor, 
self.theDisplayedEntity, self.theModelEditor.getCopyBuffer() )
        else:
            aCommand = PasteEntityPropertyList ( self.theModelEditor, self.theDisplayedEntity, self.theModelEditor.getCopyBuffer() )
        self.theModelEditor.doCommandList( [ aCommand ] )

      

    def add_new ( self ):

        # get unique name from modeleditor
        # change selection
        # call addnew in modeleditor
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return

        aType = DM_PROPERTY_STRING
        aValue = ''
        #FIXME: popup shoould take care of this

        if self.theType == 'Stepper':
            newID = self.theModelEditor.getUniqueStepperPropertyName ( self.theDisplayedEntity )
            aCommand = CreateStepperProperty( self.theModelEditor, 
self.theDisplayedEntity, newID, aValue, aType)
        else:
            newID = self.theModelEditor.getUniqueEntityPropertyName ( self.theDisplayedEntity )
            fpn = self.theDisplayedEntity + ':' + newID
            aCommand = CreateEntityProperty ( self.theModelEditor, fpn, aValue, aType)

        self.theSelection = [ newID ]
        self.theModelEditor.doCommandList( [ aCommand ] )

        # open for edit
        self.__selectRows( [ newID ], True )


    def delete ( self ):
        if len( self.theSelection ) == 0:
            return
        if self.theType == 'Stepper':
            aCommand = DeleteStepperPropertyList( self.theModelEditor, 
self.theDisplayedEntity, self.theSelection )
        else:
            aCommand = DeleteEntityPropertyList ( self.theModelEditor, self.getFullPNList() )
            
        self.theModelEditor.doCommandList( [ aCommand ] )
    



    def rename ( self, newName, anIter ):

        # if nothing changed make nothing
        #newSelection = self.__getSelection()
        #wholePropertyList = self.getPropertyList()
        #changedID = newName
        #for aSelection in newSelection:
        #   if aSelection not in wholePropertyList:
        #       changedID = aSelection
        #       break
        if not self.theModelEditor.theRuntimeObject.checkState( ME_DESIGN_MODE ):
            return
        changedID = self.theListStore.get_value( anIter, 0 )
        if changedID == newName:
            return
        if not isIDEligible( newName ):
            self.theModelEditor.printMessage( "Only alphanumeric characters and _ are allowed in fullid names!", ME_ERROR )
            self.theListStore.set_value( anIter, 0, changedID )
            return
        if self.theType == 'Stepper':
            aCommand = RenameStepperProperty( self.theModelEditor, self.theDisplayedEntity, changedID, newName )
        else:
            aCommand = RenameEntityProperty( self.theModelEditor, self.theDisplayedEntity, changedID, newName )


        if aCommand.isExecutable():
            self.theSelection = [ newName ]
            
            self.theModelEditor.doCommandList( [ aCommand ] )
            self.__changeColor()
        else:
            self.theModelEditor.printMessage( "%s cannot be renamed to %s"%(changedID, newName), ME_ERROR )
            self.theListStore.set_value( anIter, 0, changedID )
            


    def changeValue( self, newValue, anIter ):
        
        aName = self.theListStore.get_value( anIter, 0 )

        oldValue = self.getPropertyValue( aName )
        if str(oldValue) == str(newValue):
            return
        if self.theModelEditor.getMode() == ME_RUN_MODE:
            #try:
            if self.theType == "Stepper":
                self.theModelEditor.getModel().setEntityProperty(self.theDisplayedEntity,aName,newValue )
            else:
                self.theModelEditor.getModel().setEntityProperty( self.theDisplayedEntity+':'+aName, newValue )
            #except:
            #    pass
            self.update()
            return
        if self.theType == 'Stepper':
            
            aCommand = ChangeStepperProperty( self.theModelEditor, self.theDisplayedEntity, aName, newValue )
        else:
            fpn = self.theDisplayedEntity + ':' + aName
            aCommand = ChangeEntityProperty( self.theModelEditor, fpn, newValue )

        if aCommand.isExecutable():
            self.theModelEditor.doCommandList( [ aCommand ] )
            self.__changeColor()
        else:
            self.theModelEditor.printMessage( "Illegal value for %s"%aName, ME_ERROR )
            self.theListStore.set_value( anIter, 1, oldValue )
            
        

    def edit( self ):
        """
        if type is string, float or integer, select for edit
        if type is other open property editor
        get value and set property
        """
        if self.theModelEditor.isRunning():
            return
        aName = self.theSelection[0]
        if self.isEditable( aName ):
            self.__selectRows( [ aName ], gtk.FALSE, gtk.TRUE )
        else:
            aPropertyEditor = PropertyEditor( aName, self.getPropertyType( aName), self.getPropertyValue( aName ), self )

            result = aPropertyEditor.return_result()

            if result == None:
                return
            self.changeValue( result, self.__getIter( aName ) )
        


    def isEditable( self, aName ):
        if aName in [ ME_PROCESS_VARREFLIST, ME_STEPPERID ]:
            return False
        if not self.getPropertySettable( aName):
            return False
        if self.getPropertyType( aName ) not in [ DM_PROPERTY_STRING, DM_PROPERTY_FLOAT, DM_PROPERTY_INTEGER]:
            return False
        return True



    #########################################
    #    Private methods/Signal Handlers    #
    #########################################

    def __setDeleteButton(self,*args):
        selected = self.__getSelection()
        canDelete=[]
        if len(selected)>0:
            for each in selected:
                canDelete.append(self.getPropertyDeleteable(each))
        if len(canDelete)>0:
            for d in canDelete:
                if d==0:
                    self.delButton.set_sensitive(gtk.FALSE)
                    return
        self.delButton.set_sensitive(gtk.TRUE)
                

    def __button_pressed( self, *args ):
        self.__setDeleteButton()
        # when any button is pressed on list
        self.theModelEditor.setLastUsedComponent( self )        
        if args[1].button == 3:
            self.selectByUser()            
            self.theModelEditor.createPopupMenu( self, args[1] )
            return gtk.TRUE
        



    def __cursor_changed( self, *args ):
        # when row is selected in list
        if self.noActivate:
            return

        self.selectByUser()
        self.theModelEditor.setLastUsedComponent( self )



    def __add_clicked( self, *args ):
        self.add_new()       
        
    def __delete_clicked( self, *args ):
        self.delete()



    def __buildList( self ):
        """
        clear and build list
        """
        self.noActivate = True
        self.theListSelection.unselect_all()

        self.theListStore.clear()
        if self.theDisplayedEntity == None:
            return
        aValueList = []
        
        for anID in self.getPropertyList():
            aValue = []
            aValue.append( anID )
            propValue = self.getPropertyValue( anID )
            if propValue != None:
                aValue.append( propValue )
                aValue.append( self.getPropertySettable( anID ) )
                aValue.append( self.getPropertyDeleteable( anID ) )
                aValue.append( self.getPropertyDeleteable( anID ) and self.canAddNewProperty() )
                aValue.append( self.isEditable( anID ) )
                aValue.append( self.getPropertyLoadable( anID ) )
                aValue.append( self.getPropertySaveable( anID ) )
                aValue.append( self.getPropertyType( anID ))

                if self.getPropertyChanged( anID ) and self.getPropertySaveable( anID ):
                    aValue.append( PROP_SAVED )
                elif self.getPropertySaveable( anID ):
                    aValue.append( PROP_CAN_BE_SAVED )
                else:
                    aValue.append( PROP_WILL_NOT_SAVED )

            else:
                aValue += [ "N/A", False, False, False, False, False, False, "N/A", PROP_WILL_NOT_SAVED ]
            aValueList.append( aValue )
        
        
        self.__addRows( aValueList )

        self.noActivate = False

    
    def __addRows( self, aValueList ):
        """
        in: list of  [Name, Value, Settable, Creator Flag]
        """
        
        for aValue in aValueList:
            anIter = self.theListStore.append(  )
            self.theListStore.set_value ( anIter, 0 , aValue[0] )
            self.theListStore.set_value ( anIter, 1 , aValue[1] )
            self.theListStore.set_value ( anIter, 2 , aValue[2] )
            self.theListStore.set_value ( anIter, 3 , aValue[3] )
            self.theListStore.set_value ( anIter, 4 , aValue[4] )
            self.theListStore.set_value ( anIter, 5 , aValue[5] )
            self.theListStore.set_value ( anIter, 6 , aValue[6] )
            self.theListStore.set_value ( anIter, 7 , aValue[7] )
            self.theListStore.set_value ( anIter, 8 , aValue[8] )
            self.theListStore.set_value( anIter, 9 , aValue[9] )

    def __deleteRows( self, aNameList ):
        """
        in: list of string aNameList
        """

        #anIter = self.theListStore.get_iter_first()
        #while anIter != None:
        #   aName = self.theListStore.get_value( anIter, 0 )
        #   if aNameList.__contains__( aName ):
        #       self.theListStore.remove( anIter )
        #   anIter = self.theListStore.iter_next( anIter )
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



    def __selectRows( self, aNameList, forNameEdit = gtk.FALSE , forValueEdit = gtk.FALSE ):
        """
        in: list of string aNameList
            bool forEdit can only go edit mode if only one name is in namelist
        """
        self.theListSelection.unselect_all()

        if len( aNameList ) == 0:
            return
        elif len( aNameList ) == 1:
            anIter = self.theListStore.get_iter_first()
            while anIter != None:
                aName = self.theListStore.get_value (anIter, 0)
                if aNameList.__contains__( aName ):
                    self.theListSelection.select_iter( anIter )

                    aPath = self.theListStore.get_path ( anIter )
                    self.noActivate = True
                    if forValueEdit:
                        self['theTreeView'].set_cursor( aPath, self.theValueColumn, forValueEdit )
                    elif forNameEdit:
                        self['theTreeView'].set_cursor( aPath, self.theNameColumn, forNameEdit )
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
        #PLEASE, CODE IT!!!

        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            listName = self.theListStore.get_value (anIter, 0)
            if listName == aName:
                break
            anIter = self.theListStore.iter_next( anIter )
        return anIter
        
    def __updateList( self ):
        # only in RUN MODE
        anIter = self.theListStore.get_iter_first()
        while anIter != None:
            listName = self.theListStore.get_value (anIter, 0)
            listValue = self.theListStore.get_value( anIter, 1)
            try:
                listValue = float( listValue )            
            except:
                pass
            else:
                fullPN = self.theDisplayedEntity + ':' + listName
                newListValue = self.getPropertyValue( fullPN )
                if newListValue == None:
                    return
                if listValue != newListValue:
                    self.theListStore.set_value( anIter, 1, newListValue )
            anIter = self.theListStore.iter_next( anIter )

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
        self.rename ( newName, anIter )


    def __valueEdited( self, *args ):
        """
        args[0]: cellrenderer
        args[1]: path
        args[2]: newstring
        """
        
        newName = args[2]
        aPath = args[1]
        anIter = self.theListStore.get_iter_from_string( aPath )
        self.changeValue ( newName, anIter )
        

    def __changeColor(self):
        aName=self.theSelection[0]
        anIter=self.__getIter(aName)
        self.theListStore.set_value(anIter, 9,'blue')
