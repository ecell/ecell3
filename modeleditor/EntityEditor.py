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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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

import os
import os.path

from ModelEditor import *
from PropertyList import *
from ViewComponent import *
from Constants import *
from EntityCommand import *
from ShapePropertyComponent import *
from ResizeableText import *


class EntityEditor(ViewComponent):

    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach, anEntityType, aThirdFrame=None):
        

        # call superclass
        ViewComponent.__init__( self,  pointOfAttach, 'attachment_box' , 'ObjectEditor.glade' )
        

        self.theParentWindow = aParentWindow
        self.theModelEditor = self.theParentWindow.theModelEditor
        self.theType = anEntityType
        
        self.updateInProgress = False
        self.theInfoBuffer = gtk.TextBuffer()
        self.theDescriptionBuffer = gtk.TextBuffer()
        self['classname_desc'].set_buffer( self.theDescriptionBuffer )
        self['user_info'].set_buffer( self.theInfoBuffer )

        # add handlers
        self.addHandlers({
                'on_combo-entry_changed' : self.__change_class,
                'on_editor_notebook_switch_page' : self.__select_page,
                'on_ID_entry_editing_done' : self.__change_name,
                'on_user_info_move_focus' : self.__change_info
                })


        # initate Editors
        self.thePropertyList = PropertyList(self.theParentWindow, self['PropertyListFrame'] )
        aNoteBook=ViewComponent.getWidget(self,'editor_notebook')

        if aThirdFrame != None:
            #Add the ShapePropertyComponent
            aShapeFrame=gtk.VBox()
            aShapeFrame.show()
            aShapeLabel=gtk.Label('ShapeProperty')
            aShapeLabel.show()
            aNoteBook.append_page(aShapeFrame,aShapeLabel)
            self.theShapeProperty = ShapePropertyComponent(self.theParentWindow, aShapeFrame )
            self.thePropertyList.hideButtons()
        else:
            aNoteBook.set_tab_pos( gtk.POS_TOP )
            desc_vertical = self['desc_vertical']
            desc_horizontal = self['desc_horizontal']
            infoDesc = self['info_desc']
            proplist = self['PropertyListFrame']
            vertical = self['vertical_holder']
            horizontal = self['horizontal_holder']
            horizontal.remove( proplist)
            desc_horizontal.remove( infoDesc )
            vertical.pack_end( proplist)
            vertical.child_set_property( proplist, "expand", gtk.TRUE )
            vertical.child_set_property( proplist, "fill", gtk.TRUE )
            vertical.child_set_property( horizontal, "expand", gtk.FALSE )
            desc_vertical.pack_end( infoDesc )
            vertical.show_all()
            desc_vertical.show_all()
            
        
        # make sensitive change class button for process
        if self.theType == ME_PROCESS_TYPE:
            self['class_combo'].set_sensitive( gtk.TRUE )
                
        self.setDisplayedEntity( None )
                

    def getShapeProperty( self ):
        return self.theShapeProperty
                     
    def close( self ):
        """
        closes subcomponenets
        """
        self.thePropertyList.close()
        ViewComponent.close(self)


    def getDisplayedEntity( self ):
        """
        returns displayed entity
        """
        return self.theDisplayedEntity

    def bringToTop( self ):
        self['ID_entry'].grab_focus()

    def update ( self ):
        """
        """
        # update Name
        if self.theModelEditor.getMode() == ME_DESIGN_MODE:
            self.updateEditor()
        if self.theDisplayedEntity != self.thePropertyList.getDisplayedEntity():
            self.thePropertyList.setDisplayedEntity( self.theDisplayedEntity)
        else:
            # update propertyeditor
            self.thePropertyList.update()


    def updateEditor( self ):
        self.updateInProgress = True
        if self.theDisplayedEntity !=None:
            nameText = self.theDisplayedEntity.split(':')[2]
        else:
            nameText = ''
        self['ID_entry'].set_text( nameText )

        sensitiveFlag = gtk.FALSE
        if self.theDisplayedEntity != None:
            sensitiveFlag = gtk.TRUE
        self['user_info'].set_sensitive( sensitiveFlag )
        if sensitiveFlag and self.theDisplayedEntity == ME_ROOTID:
            sensitiveFlag = gtk.FALSE
        self['ID_entry'].set_sensitive( sensitiveFlag )
        

        # delete class list from combo
        self['class_combo'].entry.set_text('')
        self['class_combo'].set_popdown_strings([''])
        self['class_combo'].set_sensitive( gtk.FALSE )
        self['class_combo'].set_data( 'selection', '' )
        descText = ''

        if self.theDisplayedEntity != None:
            # get actual class
            actualclass = self.theModelEditor.getModel().getEntityClassName( self.theDisplayedEntity )

            if self.theDisplayedEntity.split(':')[0] == ME_PROCESS_TYPE:

                self['class_combo'].set_sensitive( gtk.TRUE )
                
            # get class list
            classStore = copyValue ( self.theModelEditor.getDMInfo().getClassList( ME_PROCESS_TYPE) )

            self['class_combo'].set_popdown_strings(classStore)
        
            # select class
            self['class_combo'].entry.set_text( actualclass )
            self['class_combo'].set_data( 'selection', actualclass )
            descText = self.theModelEditor.getDMInfo().getClassInfo( actualclass, DM_DESCRIPTION )

        self.__setDescriptionText( descText )

        # update syspath if apropriate
        syspathText = ''
        if self.theDisplayedEntity !=None:
            syspathText = self.theDisplayedEntity.split(':')[1]

        self['path_entry'].set_text( syspathText )

        infoText = ''
        # FIXME, get this from DMINFO!
        if self.theDisplayedEntity != None :
            infoText = self.theModelEditor.theModelStore.getEntityInfo( self.theDisplayedEntity )
        self.__setInfoText( infoText )
        self.updateInProgress = False




    def setDisplayedEntity ( self, selectedID ):
        """
        """

        self.theDisplayedEntity = selectedID 

        # sets displayed entity for Property Editor
        self.thePropertyList.setDisplayedEntity('Entity', self.theDisplayedEntity )
    
        self.updateEditor()



    def addLayoutEditor( self, aLayoutEditor ):
        pass



    def changeClass( self, newClass ):
        currentClass = self.theModelEditor.getModel().getEntityClassName( self.theDisplayedEntity )
        if currentClass == newClass:
            return
        aCommand = ChangeEntityClass( self.theModelEditor, newClass, self.theDisplayedEntity )
        self.theModelEditor.doCommandList( [ aCommand ] )



    def changeName ( self, newName ):
        
        newTuple = self.theDisplayedEntity.split(':')
        oldName=newTuple[2]
        newTuple[2] = newName
        newID = ':'.join( newTuple )
        sysPath = self.theDisplayedEntity.split(':')[1]
        existEntityList = self.theModelEditor.getModel().getEntityList(newTuple[0],sysPath)
        if not isIDEligible( newName ):
            self.theModelEditor.printMessage( "Only alphanumeric characters and _ are allowed in entity names!", ME_ERROR )

        
        elif not newName in existEntityList:
            aCommand = RenameEntity( self.theModelEditor, self.theDisplayedEntity, newID )
            if aCommand.isExecutable:
                #ResizeableText.deregisterText(oldName)
                self.theDisplayedEntity = newID
                self.theModelEditor.doCommandList ( [ aCommand ] )
                self.theParentWindow.selectEntity( [ newID ] )
            else:
                self.theModelEditor.printMessage( "%s cannot be renamed to %s"%(self.theDisplayedEntity, newID ), ME_ERROR )
        self.updateEditor()

    def changeInfo( self, newInfo ):
        aCommand = SetEntityInfo( self.theModelEditor, self.theDisplayedEntity, newInfo )
        self.theModelEditor.doCommandList( [ aCommand ] )



    #########################################
    #    Private methods/Signal Handlers    #
    #########################################


    
    def __change_class( self, *args ):
        """
        called when class is to be changed
        """
        if args[0].get_text() == '':
            return
        if self.updateInProgress:
            return
        newClass = self['class_combo'].entry.get_text()
        self.changeClass( newClass )



    def __select_page( self, *args ):
        """
        called when editor pages are selected
        """
        pass


    def __change_name ( self, *args ):
        if self.updateInProgress:
            return
        newName = self['ID_entry'].get_text()
        self.changeName( newName )
        
    def __change_info( self, *args ):
        if self.updateInProgress:
            return
        newInfo = self.__getInfoText()
        self.changeInfo( newInfo )


    def __setDescriptionText( self, textString ):
        self.theDescriptionBuffer.set_text( textString )


    def __getInfoText( self ):
        endIter = self.theInfoBuffer.get_end_iter()
        startIter = self.theInfoBuffer.get_start_iter()
        return self.theInfoBuffer.get_text( startIter, endIter, gtk.TRUE )


    def __setInfoText( self, textString ):
        self.theInfoBuffer.set_text( textString )

