#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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
#'Programming: Thaw Tint' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

import os
import os.path

import gtk

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.ViewComponent import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapePropertyComponent import *
from ecell.ui.model_editor.LinePropertyComponent import *

class  LinePropertyComponent(ViewComponent):
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow,aConnObjectEditor, anObject ,pointOfAttach ):
        
        
        self.theModelEditor = aParentWindow.theModelEditor
        self.theParent  =aParentWindow
        self.theConnObjectEditorWindow = aConnObjectEditor
        
        ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'LinePropertyComponent.glade' )


        #Add Handlers

        self.addHandlers({
            'on_arw_color_button_clicked'   : self.__ColorFillSelection_displayed,
            'on_arw_style_changed'  : self.__change_arw_style,
            'on_arw_type_selection_received':self.__change_arw_type,
            })

        self.theColor=None
        self.theColorDialog=None
        self.theDrawingArea=None
        self.theColorList=['red', 'yellow', 'green', 'brown', 'blue', 'magenta',
                 'darkgreen', 'bisque1']
        self.theAvailableArrowTypeList=[]
        self.theArwTypeCombo=ViewComponent.getWidget(self,'cboArwType')
        self.theLineTypeCombo=ViewComponent.getWidget(self,'cboArwStyle')
        self.setDisplayedLineProperty( anObject )
               
    def close( self ):
        """
        closes subcomponenets
        """
        ViewComponent.close(self)

    #########################################
    #    Private methods/Signal Handlers    #
    #########################################   

    def __ColorFillSelection_displayed( self,*args):
        aDialog,aColorSel=self.setColorDialog()
        response=aDialog.run()
        if response==gtk.RESPONSE_OK:
            aColor=aColorSel.get_current_color()
            aDa=ViewComponent.getWidget(self,'da_fill')
            aDa.modify_bg(gtk.STATE_NORMAL,aColor)
            aDialog.destroy()
            self.setHexadecimal(aColor,OB_FILL_COLOR)
            self.__LineProperty_updated(OB_FILL_COLOR,self.theHex)
        else:
            aDialog.destroy()

    def __LineProperty_updated(self, aKey,aNewValue):
        self.theConnObjectEditorWindow.modifyConnObjectProperty(aKey,aNewValue)     


    def __change_arw_style( self, *args ):
        newClass = (ViewComponent.getWidget(self,'arw_style')).get_text()
        
    def __change_arw_type( self, *args ):
        newArwType = (ViewComponent.getWidget(self,'arw_type')).get_text()
        self.changeArwType(newArwType)
        

    ###############################################################################
    #PRIVATE METHOD#
    ###############################################################################
    def populateComboBox(self,anObject):
        self.theAvailableArrowTypeList=anObject.getAvailableArrowType()
        anObjArrowType = anObject.getProperty(OB_SHAPE_TYPE)
        for i in range (len(self.theAvailableArrowTypeList)):
            if self.theAvailableArrowTypeList[i]==anObjArrowType:
                temp = self.theAvailableArrowTypeList[0]
                self.theAvailableArrowTypeList[0]=self.theAvailableArrowTypeList[i]
                self.theAvailableArrowTypeList[i]=temp
        self.theArwTypeCombo.set_popdown_strings(self.theAvailableArrowTypeList)
        self.theLineTypeCombo.set_popdown_strings(anObject.getAvailableLineType())

    def setDisplayedLineProperty(self ,connObj):
        if len(connObj.values()) == 0:
            self.clearLineProperty()
            return
        self.theObject=connObj.values()[0]
        self.theFillColor=self.getColorObject(self.theObject.getProperty(OB_FILL_COLOR))
        self.thecoef = self.theObject.getProperty(CO_COEF)
        self.populateComboBox(self.theObject)
        self.updateLineProperty()

    def clearLineProperty(self):
        self.theFillColor = self.getColorObject([0,0,0])
        self.thecoef = 0
        self.theArwTypeCombo.set_sensitive(True)
        self.theLineTypeCombo.set_sensitive(True)
        ViewComponent.getWidget(self,'arw_color_button').set_sensitive(True)
        ViewComponent.getWidget(self,'chk_last_arw').set_active(False)
        ViewComponent.getWidget(self,'chk_first_arw').set_active(False)

    def updateLineProperty(self):
        self.theArwTypeCombo.set_sensitive(True)
        self.theLineTypeCombo.set_sensitive(True)
        ViewComponent.getWidget(self,'arw_color_button').set_sensitive(True)
        ViewComponent.getWidget(self,'da_fill').modify_bg(gtk.STATE_NORMAL,self.theFillColor)
        
        if int(self.thecoef) < 0:
            ViewComponent.getWidget(self,'chk_first_arw').set_active(True)
            ViewComponent.getWidget(self,'chk_last_arw').set_active(False)
        elif int(self.thecoef) > 0:
            ViewComponent.getWidget(self,'chk_last_arw').set_active(True)
            ViewComponent.getWidget(self,'chk_first_arw').set_active(False)
        elif int(self.thecoef) == 0:
            ViewComponent.getWidget(self,'chk_last_arw').set_active(False)
            ViewComponent.getWidget(self,'chk_first_arw').set_active(False)
        
    def changeArwType(self, newArwType):
        if newArwType!="" and  self.theObject.getProperty(OB_SHAPE_TYPE) != newArwType:
            self.__LineProperty_updated(OB_SHAPE_TYPE,newArwType)
                
    
    def setColorDialog( self, *args ):
        aColor=gtk.gdk.color_parse(self.theColorList[randint (0, 3)])
        aDialog=gtk.ColorSelectionDialog("Select Fill Color")
        aColorSel=aDialog.colorsel
        aColorSel.set_previous_color(aColor)
        aColorSel.set_current_color(aColor)
        aColorSel.set_has_palette(True)
        return[aDialog,aColorSel]
    
    def setHexadecimal(self,theColor,theColorMode ):
        self.theRed=theColor.red
        self.theGreen=theColor.green
        self.theBlue=theColor.blue
        self.theHex = [self.theRed,self.theGreen,self.theBlue]
        

    def getColorObject(self, anRGB):
        aColor=gtk.gdk.color_parse(self.theColorList[randint (0, 3)])
        aColor.red= anRGB[0]
        aColor.green= anRGB[1]
        aColor.blue= anRGB[2]
        return aColor
    






