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
        self.theColorList=['red', 'yellow', 'green', 'brown', 'blue', 'magenta',
                 'darkgreen', 'bisque1']
        self.theCombo = ViewComponent.getWidget(self,'shape_combo')
        self.noUpdate = True
        self.theObjectDict = {}

        self.theFillColor = None
        self.theOutlineColor = None

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
          
        self.modifyObjectProperty(aKey,aNewValue)
        
        

            
    #########################################
    #    Private methods                #
    #########################################

    
        
    def setDisplayedShapeProperty(self ,anObjectDict, aType):
        self.theObjectDict = anObjectDict
        self.theType = aType

        self.populateComboBox()
        if len( self.theObjectDict.values()) > 0:
            theObject = self.theObjectDict.values()[0]
            self.theFillColor=self.getColorObject( theObject.getProperty(OB_FILL_COLOR) )
            self.theOutlineColor=self.getColorObject( theObject.getProperty(OB_OUTLINE_COLOR) )
        else:
            self.theFillColor=self.getColorObject([65535, 65535, 65535])
            self.theOutlineColor=self.getColorObject([0,0,0])

        self.updateShapeProperty()


    def updateShapeProperty(self):
        if len( self.theObjectDict.values() ) > 0:
            colorSettable = gtk.TRUE
        else:
            colorSettable = gtk.FALSE
        heightSensitive = gtk.FALSE
        widthSensitive = gtk.FALSE
        labelSensitive = gtk.FALSE
        label = ""
        x = ""
        y = ""
        if len( self.theObjectDict.keys() ) == 1:
            theObject = self.theObjectDict.values()[0]
            if theObject.getProperty( OB_HASFULLID ):
                self.theFullId = theObject.getProperty(OB_FULLID)
                label = self.theFullId.split(':')[2]
                if theObject.getProperty(OB_TYPE) == OB_TYPE_SYSTEM:
                    heightSensitive = gtk.TRUE
                    widthSensitive = gtk.TRUE
            else:
                heightSensitive = gtk.TRUE
                widthSensitive = gtk.TRUE
                labelSensitive = gtk.TRUE

            
            x = str(theObject.getProperty( OB_DIMENSION_X ))
            y = str(theObject.getProperty( OB_DIMENSION_Y ))
        
        ViewComponent.getWidget(self,'ent_label').set_text(label )
        ViewComponent.getWidget(self,'ent_height').set_text( y )
        ViewComponent.getWidget(self,'ent_width').set_text( x )
        ViewComponent.getWidget(self,'FillButton').set_sensitive(colorSettable)
        ViewComponent.getWidget(self,'OutlineButton').set_sensitive(colorSettable)
        ViewComponent.getWidget(self,'ent_label').set_sensitive(labelSensitive)
        ViewComponent.getWidget(self,'ent_width').set_sensitive(widthSensitive)
        ViewComponent.getWidget(self,'ent_height').set_sensitive(heightSensitive)
        
        ViewComponent.getWidget(self,'da_fill').modify_bg(gtk.STATE_NORMAL,self.theFillColor)
        ViewComponent.getWidget(self,'da_out').modify_bg(gtk.STATE_NORMAL,self.theOutlineColor)


        
    def populateComboBox(self):
        # populate list item of combobox
        if self.theType not in [ "None", "Mixed" ]:
            comboSensitive = gtk.TRUE
            theObject = self.theObjectDict.values()[0]
            aShapeList = theObject.getAvailableShapes()
            currentShape = theObject.getProperty(OB_SHAPE_TYPE)
            aShapeList.remove( currentShape )
            aShapeList.insert(0, currentShape )
        else:
            aShapeList = []
            comboSensitive = gtk.FALSE
        self.noUpdate = True
        self.theCombo.set_popdown_strings( aShapeList )
        self.noUpdate = False
        ViewComponent.getWidget(self,'combo_entry').set_sensitive( comboSensitive)
         

    def setColorDialog( self, *args ):
        aColor=gtk.gdk.color_parse(self.theColorList[randint (0, 3)])
        aDialog=gtk.ColorSelectionDialog("Select Fill Color")
        aColorSel=aDialog.colorsel
        aColorSel.set_previous_color(aColor)
        aColorSel.set_current_color(aColor)
        aColorSel.set_has_palette(gtk.TRUE)
        return[aDialog,aColorSel]
        
    def setHexadecimal(self,theColor,theColorMode ):
        self.theHex = [theColor.red,theColor.green,theColor.blue]
        

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
    

    # ==========================================================================
    def modifyObjectProperty(self,aPropertyName, aPropertyValue):
        if len( self.theObjectDict.values() ) == 0:
            raise "There is no object, cannot change property!"
        self.theLayout = self.theObjectEditorWindow.theLayout
        if  aPropertyName == OB_OUTLINE_COLOR  or aPropertyName == OB_FILL_COLOR or aPropertyName == OB_SHAPE_TYPE :
            # create commandList
            aCommandList = []
            for anObjectID in self.theObjectDict.keys():
                aCommandList.append( SetObjectProperty(self.theLayout, anObjectID, aPropertyName, aPropertyValue )  )

            self.theLayout.passCommand( aCommandList )
            
        
        #elif aPropertyName == OB_FULLID:
        #    if self.theObject.getProperty(OB_HASFULLID):
        #        if isFullIDEligible( aPropertyValue ):
        #            aCommand = RenameEntity( self.theModelEditor, self.theObject.getProperty(OB_FULLID), aPropertyValue )
        #            if aCommand.isExecutable():
        #                self.theModelEditor.doCommandList( [ aCommand ] )
        #        else:
        #            self.theModelEditor.printMessage( "Only alphanumeric characters and _ are allowed in fullid names!", ME_ERROR )

        #        self.selectEntity( [self.theObject] )
                            

        #    else:
        #        pass
        
        elif  aPropertyName == OB_DIMENSION_Y  :
            if len( self.theObjectDict.values() ) != 1 :
                raise "Cannot change dimensions of more than one object!"
            theObjectID, theObject = self.theObjectDict.items()[0]
            
            objHeight = theObject.getProperty(OB_DIMENSION_Y)
            deltaHeight = objHeight-aPropertyValue
            maxShiftPos= theObject.getMaxShiftPos(DIRECTION_DOWN)
            maxShiftNeg= theObject.getMaxShiftNeg(DIRECTION_DOWN) 
            if deltaHeight>0:
                if  maxShiftNeg > deltaHeight:
                    # create command
                    aCommand=ResizeObject(self.theLayout, theObjectID,0,-deltaHeight, 0, 0 )
                    self.theLayout.passCommand( [aCommand] )

            elif deltaHeight<0:
                if  maxShiftPos > -deltaHeight:
                    aCommand=ResizeObject(self.theLayout, theObjectID,0, -deltaHeight, 0, 0 )
                    self.theLayout.passCommand( [aCommand] )
                
            self.updateShapeProperty()
        
        elif  aPropertyName == OB_DIMENSION_X :
            if len( self.theObjectDict.values() ) != 1 :
                raise "Cannot change dimensions of more than one object!"
            theObjectID, theObject = self.theObjectDict.items()[0]

            objWidth = theObject.getProperty(OB_DIMENSION_X)
            deltaWidth = objWidth-aPropertyValue
            maxShiftPos= theObject.getMaxShiftPos(DIRECTION_RIGHT)
            maxShiftNeg= theObject.getMaxShiftNeg(DIRECTION_RIGHT) 
            if deltaWidth>0:
                if  maxShiftNeg > deltaWidth:
                    # create command
                    aCommand=ResizeObject(self.theLayout, theObjectID,0, 0, 0, -deltaWidth )
                    self.theLayout.passCommand( [aCommand] )

            elif deltaWidth<0:
                if  maxShiftPos > -deltaWidth:
                    # create command
                    aCommand=ResizeObject(self.theLayout, theObjectID,0, 0,0, -deltaWidth )
                    self.theLayout.passCommand( [aCommand] )
            self.updateShapeProperty()

        



