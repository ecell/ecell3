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
from Utils import *
import gtk
import gobject

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from ShapePropertyComponent import *
from LinePropertyEditor import *
from LayoutManager import *
from Layout import *
from EditorObject import *
from LayoutCommand import *
from EntityCommand import *


class ConnectionObjectEditorWindow:
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aModelEditor, aLayoutName, anObjectId ):
        """
        sets up a modal dialogwindow displaying 
        the MEVariableReferenceEditor and the LineProperty
             
        """ 
        self.theModelEditor = aModelEditor  
        
        # Create the Dialog
#        self.win = gtk.Dialog('ConnectionObject' , None)
#        self.win.connect("destroy",self.destroy)

        # Sets size and position
#        self.win.set_border_width(2)
#        self.win.set_default_size(300,75)
#        self.win.set_position(gtk.WIN_POS_MOUSE)

        # Sets title
#        self.win.set_title('ConnectionObjectEditor')
#        aPixbuf16 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
#                                os.sep + "glade" + os.sep + "modeleditor.png")
#        aPixbuf32 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
#                                os.sep + "glade" + os.sep + "modeleditor32.png")
#        self.win.set_icon_list(aPixbuf16, aPixbuf32)

        self.theTopFrame = gtk.VBox()
        self.getTheObject(aLayoutName, anObjectId)
        self.theComponent = VariableReferenceEditorComponent( self, self.theTopFrame,self.theLayout,self.theObjectMap)
        self.theComponent.setDisplayedVarRef(self.theLayout,self.theObjectMap)
#        self.win.show_all()
        self.update()
        self.bringToTop()
        
    def bringToTop( self ):
        self.theModelEditor.theMainWindow.setSmallWindow( self.theTopFrame )
        self.theComponent.bringToTop()
        

    #########################################
    #    Private methods            #
    #########################################

    def setDisplayConnObjectEditorWindow(self,aLayoutName, anObjectId):
        self.getTheObject( aLayoutName, anObjectId)
        self.theComponent.setDisplayedVarRef(self.theLayout,self.theObjectMap)
        self.update()
        self.bringToTop()
                
    def getTheObject(self,aLayoutName, anObjectId):
        self.theLayout =self.theModelEditor.theLayoutManager.getLayout(aLayoutName)
        self.theObjectMap = {}
        if anObjectId == None:
            return
        elif type( anObjectId ) == type([]):
            for anId in anObjectId:
                self.theObjectMap[ anId] = self.theLayout.getObject(anId)


        elif type( anObjectId ) == str:
            self.theObjectMap[ anObjectId] = self.theLayout.getObject(anObjectId)
        
    def modifyConnObjectProperty(self,aPropertyName, aPropertyValue):
        aCommandList = []

        if  aPropertyName == OB_FILL_COLOR :
            # create command
            for anId in self.theObjectMap.keys():
                aCommandList += [ SetObjectProperty(self.theLayout, anId, aPropertyName, aPropertyValue )  ]


        if  aPropertyName == OB_SHAPE_TYPE :
            # create command
            for anId in self.theObjectMap.keys():
                aCommandList += [SetObjectProperty( self.theLayout, anId, aPropertyName, aPropertyValue ) ]
        if len( aCommandList ) > 0:
            self.theLayout.passCommand( aCommandList )


    def update(self, aType = None, aFullID = None):
        self.theComponent.update()

    # ==========================================================================
    def destroy( self, *arg ):
        """destroy dialog
        """
        pass        
#        self.win.destroy()

        

        
        

        

    


    






