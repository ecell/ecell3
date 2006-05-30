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
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#
import gtk

from ModelEditor import *
from ListWindow import *
import os
import os.path
import string
from SystemTree import *
from EntityList import *
from EntityEditor import *
from Utils import *


class EntityListTab(ListWindow):


    def __init__( self, aModelEditor,aRoot=None ):
        """
        in: ModelEditor theModelEditor
        returns nothing
        """
        self.theModelEditor = aModelEditor

        # init superclass
        ListWindow.__init__( self, self.theModelEditor ,aRoot)
        


    def openWindow( self ):
        """
        in: nothing
        returns nothing
        """

        # superclass openwindow
        ListWindow.openWindow( self )

        # create systree, processlist, propertylist
        self.theSystemTree = SystemTree( self, self['SystemTreeFrame'] )
        self.theEntityType = self.__getEntityType()
        
        ###############################################################################
        self.theEntityList = EntityList( self, self['EntityListFrame'], self.theEntityType )
       
        self.theEntityEditorList = EntityEditor( self, self['EntityEditorFrame'], self.theEntityType )
           
        # add signal handlers
        self.addHandlers({ 'on_variable1_activate' : self.__entitychooser_changed, 
                            'on_process1_activate' : self.__entitychooser_changed
                })

        self.theEntityList.changeDisplayedType( self.theEntityType )


        self.selectEntity( [ME_ROOTID] )


        self.update()
           
        
    def setLastActiveComponent( self, aComponent ):
        self.theLastActiveComponent = aComponent
        self.updatePropertyList()


    def update( self, aType = None, aFullID = None ):
        #if self.theModelEditor.getMode() == ME_RUN_MODE:
        #    self.theEntityEditorList.thePropertyList.update()
        #    return
        if aType == ME_SYSTEM_TYPE:
            self.updateSystemTree( aFullID )
        elif aType == ME_STEPPER_TYPE:
            self.updatePropertyList()
        elif aType == self.theEntityType:
            self.updateEntityList( aFullID )
        elif aType == ME_VARIABLE_TYPE and self.theEntityType == ME_PROCESS_TYPE:
            self.updatePropertyList()
        elif aType == ME_PROPERTY_TYPE:
            self.updatePropertyList( aFullID )
        else:
            self.updateSystemTree()
        
    def selectEntity( self, anEntityList ):

        aType = getFullIDType ( anEntityList[0] )
        if aType == ME_SYSTEM_TYPE:
            self.theLastActiveComponent = self.theSystemTree
            
        elif aType in [ ME_PROCESS_TYPE, ME_VARIABLE_TYPE ]:
            displayedType = self.__getEntityType()
            #self.theEntityList.aValue = self.theEntityEditorList.getValue()
            if aType != displayedType:
                
                self.theEntityList.changeDisplayedType( aType )

            self.theLastActiveComponent = self.theEntityList 
        else:
            raise Exception("Wrong type to select %s"% aType )
        self.theLastActiveComponent.changeSelection( anEntityList )     
            
        self.theLastActiveComponent.selectByUser()
        
        

    def updateSystemTree ( self, aSystemFullID = None ):
        """
        in: string aSystemFullID where changes happened
        """

        if not self.exists():
            return
        if aSystemFullID != None:
            self.theSystemTree.update ( aSystemFullID )
        self.updateEntityList( aSystemFullID )



    def updateEntityList ( self, aFullID = None ):
        """
        in: string aFullID where changes happened
        """
        if not self.exists():
            return

        displayedFullID = self.theEntityList.getDisplayedSysID() 
        systemTreeFullIDs = self.theSystemTree.getSelectedIDs() 
        if len(systemTreeFullIDs) != 1:
            systemTreeFullID = None
        else:
            systemTreeFullID = systemTreeFullIDs[0]

        # check if new system is selected
        if displayedFullID != systemTreeFullID:
            self.theEntityList.setDisplayedSysID( systemTreeFullID )

        elif displayedFullID == aFullID or aFullID == None:
            self.theEntityList.update( )

        # check whether there were any changes in displayed data
        self.updatePropertyList( aFullID )


    
    def updatePropertyList ( self, aFullID = None ):
        """
        in: anID where changes happened
        """

        if not self.exists():
            return
        # get selected process or systemid
        selectedID = None

        selectedIDs = self.theLastActiveComponent.getSelectedIDs()
        if len(selectedIDs) == 1:
            selectedID = selectedIDs[0]
        else:
            selectedID = None
        
        # get displayed entity from propertylist
        
        propertyListEntity = self.theEntityEditorList.getDisplayedEntity()
        # check if selection was changed
        if propertyListEntity != selectedID :

            self.theEntityEditorList.setDisplayedEntity ( selectedID )

        elif aFullID == selectedID or aFullID == None or aFullID[-4:] == "SIZE":

            self.theEntityEditorList.update()

    def changeEntityType( self ):
        self.theEntityType = self.__getEntityType()
        self.theEntityList.changeDisplayedType( self.theEntityType )
        self.updateEntityList()


    #############################
    #      SIGNAL HANDLERS      #
    #############################

    def deleted( self, *arg ):
        ListWindow.deleted( self, *arg )
        self.theSystemTree.close()
        self.theEntityList.close()
        self.theEntityEditorList.close()
        self.theModelEditor.theEntityListWindowList.remove( self )
        self.theModelEditor.updateWindows()
        return True


    def __entitychooser_changed( self, *args ):
    
        self.changeEntityType()     
      

    def __getEntityType( self ):
        """
        returns string type of entity chooser
        """
        anIndex = self['EntityChooser'].get_history()
        return (ME_VARIABLE_TYPE, ME_PROCESS_TYPE)[ anIndex ]


