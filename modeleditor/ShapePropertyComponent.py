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
#'Programming: Dini Karnaga' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from whrandom import randint
from SystemObject import *
from EditorObject import *




class ShapePropertyComponent(ViewComponent):

    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self,anObjectEditorWindow,pointOfAttach ):
        
        self.theObjectEditorWindow = anObjectEditorWindow
        self.theModelEditor = self.theObjectEditorWindow.theModelEditor
        
        ViewComponent.__init__( self, pointOfAttach, 'attachment_frame', 'ShapePropertyComponent.glade' )
        
        # the variables
        self.ShapeName=''
        self.theShapeList=[]
        self.theColor=None
        self.theColorDialog=None
        self.theDrawingArea=None
        self.theColorList=['red', 'yellow', 'green', 'brown', 'blue', 'magenta',
                 'darkgreen', 'bisque1']
        self.theCombo=ViewComponent.getWidget(self,'shape_combo')
        self.noUpdate=True
        self.theObject=None
        self.theFullId=None
        self.theFillColor=None
        self.theOutlineColor=None

        # the handlers dictionary
        self.addHandlers({ 
            'on_EntityLabel_activate' : self.__EntityLabel_displayed,\
            'on_EntityLabel_focus_out' : self.__EntityLabel_displayed,\
            'on_ComboEntry_changed' : self.__ShapeType_displayed,\
            'on_EntityWidth_activate' : self.__EntityWidth_displayed,\
            'on_EntityWidth_focus_out' : self.__EntityWidth_displayed,\
            'on_EntityHeight_activate' : self.__EntityHeight_displayed,\
            'on_EntityHeight_focus_out' : self.__EntityHeight_displayed,\
            'on_FillButton_clicked' : self.__ColorFillSelection_displayed,\
            'on_OutlineButton_clicked' : self.__ColorOutSelection_displayed
            })
        
        

    #########################################
    #    Signal Handlers                    #
    #########################################
    def __EntityLabel_displayed( self, *args ):
        aShapeLabel=(ViewComponent.getWidget(self,'ent_label')).get_text()
        self.newTuple[2]=aShapeLabel
        aShapeLabel = ':'.join(self.newTuple)
        self.__ShapeProperty_updated(OB_FULLID,aShapeLabel)
        
        

    def __EntityWidth_displayed( self, *args ):
        aShapeWidth=(ViewComponent.getWidget(self,'ent_width')).get_text()
        if self.checkNumeric(aShapeWidth):
            self.__ShapeProperty_updated(OB_DIMENSION_X,float(aShapeWidth))
        else:
            self.updateShapeProperty()
        
    def __EntityHeight_displayed( self, *args ):
        aShapeHeight=(ViewComponent.getWidget(self,'ent_height')).get_text()
        if self.checkNumeric(aShapeHeight):
            self.__ShapeProperty_updated(OB_DIMENSION_Y,float(aShapeHeight))
        else:
            self.updateShapeProperty()
        
    def __ShapeType_displayed( self, *args ):
        if self.noUpdate:
            return
        aComboEntryWidget=ViewComponent.getWidget(self,'combo_entry')
        
        aShapeType=aComboEntryWidget.get_text()
        if aShapeType == '':
            return
        self.__ShapeProperty_updated(OB_SHAPE_TYPE,aShapeType)
            
        
        
    def __ColorFillSelection_displayed( self,*args):
        aDialog,aColorSel=self.setColorDialog()
        response=aDialog.run()
        if response==gtk.RESPONSE_OK:
            aColor=aColorSel.get_current_color()
            aDa=ViewComponent.getWidget(self,'da_fill')
            aDa.modify_bg(gtk.STATE_NORMAL,aColor)
            aDialog.destroy()
            self.setHexadecimal(aColor,OB_FILL_COLOR)
            self.__ShapeProperty_updated(OB_FILL_COLOR,self.theHex)
        else:
            aDialog.destroy()
            

    def __ColorOutSelection_displayed( self,*args):
        aDialog,aColorSel=self.setColorDialog()
        response=aDialog.run()
        if response==gtk.RESPONSE_OK:
            aColor=aColorSel.get_current_color()
            aDa=ViewComponent.getWidget(self,'da_out')
            aDa.modify_bg(gtk.STATE_NORMAL,aColor)
            aDialog.destroy()
            self.setHexadecimal(aColor,OB_OUTLINE_COLOR)
            self.__ShapeProperty_updated(OB_OUTLINE_COLOR,self.theHex)
        else:
            aDialog.destroy()
    
    def __ShapeProperty_updated(self, aKey,aNewValue):
          
        self.theObjectEditorWindow.modifyObjectProperty(aKey,aNewValue)
        self.update=True
        

    def __ShapeProperty_displayed(self,*args):
        keys=self.theShapeDic.keys()
        keys.sort()
        for aKey in keys:
            self.theModelEditor.printMessage(aKey + ':' + str(self.theShapeDic[aKey]),ME_PLAINMESSAGE)
            
        self.theModelEditor.printMessage('',ME_PLAINMESSAGE)
        
        

            
    #########################################
    #    Private methods                #
    #########################################

    
        
    def setDisplayedShapeProperty(self ,anObject):
        if anObject==None:
            self.clearShapeProperty()
            return
        self.theObject=anObject
        self.populateComboBox()
        self.theFillColor=self.getColorObject(self.theObject.getProperty(OB_FILL_COLOR))
        self.theOutlineColor=self.getColorObject(self.theObject.getProperty(OB_OUTLINE_COLOR))
        self.updateShapeProperty()


    def updateShapeProperty(self):
        ViewComponent.getWidget(self,'combo_entry').set_sensitive(gtk.TRUE)
        ViewComponent.getWidget(self,'FillButton').set_sensitive(gtk.TRUE)
        ViewComponent.getWidget(self,'OutlineButton').set_sensitive(gtk.TRUE)
        if self.theObject.getProperty(OB_HASFULLID):
            self.theFullId=self.theObject.getProperty(OB_FULLID)
            self.newTuple = self.theFullId.split(':')
            label = self.theFullId.split(':')[2]
            ViewComponent.getWidget(self,'ent_label').set_sensitive(gtk.FALSE)
            ViewComponent.getWidget(self,'ent_width').set_sensitive(gtk.FALSE)
            ViewComponent.getWidget(self,'ent_height').set_sensitive(gtk.FALSE)
            
        else:
            label = ''
            ViewComponent.getWidget(self,'ent_label').set_sensitive(gtk.TRUE)
            ViewComponent.getWidget(self,'ent_width').set_sensitive(gtk.TRUE)
            ViewComponent.getWidget(self,'ent_height').set_sensitive(gtk.TRUE)
        
        if self.theObject.getProperty(OB_TYPE) == OB_TYPE_SYSTEM:
            ViewComponent.getWidget(self,'ent_width').set_sensitive(gtk.TRUE)
            ViewComponent.getWidget(self,'ent_height').set_sensitive(gtk.TRUE)
        
        ViewComponent.getWidget(self,'ent_label').set_text(label )
        ViewComponent.getWidget(self,'ent_height').set_text( str(self.theObject.getProperty( OB_DIMENSION_Y ) ))
        ViewComponent.getWidget(self,'ent_width').set_text( str(self.theObject.getProperty( OB_DIMENSION_X )) )
        
        ViewComponent.getWidget(self,'da_fill').modify_bg(gtk.STATE_NORMAL,self.theFillColor)
        ViewComponent.getWidget(self,'da_out').modify_bg(gtk.STATE_NORMAL,self.theOutlineColor)


    def clearShapeProperty(self):
        ViewComponent.getWidget(self,'ent_label').set_text('' )
        ViewComponent.getWidget(self,'ent_height').set_text( '')
        ViewComponent.getWidget(self,'ent_width').set_text('')
        ViewComponent.getWidget(self,'combo_entry').set_sensitive(gtk.FALSE)
        self.theFillColor=self.getColorObject([65535, 65535, 65535])
        self.theOutlineColor=self.getColorObject([0,0,0])
        ViewComponent.getWidget(self,'da_fill').modify_bg(gtk.STATE_NORMAL,self.theFillColor)
        ViewComponent.getWidget(self,'da_out').modify_bg(gtk.STATE_NORMAL,self.theOutlineColor)
        ViewComponent.getWidget(self,'FillButton').set_sensitive(gtk.FALSE)
        ViewComponent.getWidget(self,'OutlineButton').set_sensitive(gtk.FALSE)
        

        
    def populateComboBox(self):
        # populate list item of combobox
        aShapeList=self.theObject.getAvailableShapes()
        for i in range ( len( aShapeList ) ):
            if aShapeList[i] == self.theObject.getProperty(OB_SHAPE_TYPE):
                temp = aShapeList[0]
                aShapeList[0] = aShapeList[i]
                aShapeList[i]=temp
                break
        self.noUpdate = True
        self.theCombo.set_popdown_strings( aShapeList )
        self.noUpdate = False

         

    def setColorDialog( self, *args ):
        aColor=gtk.gdk.color_parse(self.theColorList[randint (0, 3)])
        aDialog=gtk.ColorSelectionDialog("Select Fill Color")
        aColorSel=aDialog.colorsel
        aColorSel.set_previous_color(aColor)
        aColorSel.set_current_color(aColor)
        aColorSel.set_has_palette(gtk.TRUE)
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
    
    def checkNumeric(self, aNumber):
        try:
            eval(aNumber)
            return True
        except:
            return False
    




