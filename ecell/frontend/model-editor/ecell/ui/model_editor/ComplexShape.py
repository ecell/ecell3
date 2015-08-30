#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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

import os
import gtk
import gtk.gdk
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas

import ecell.ui.model_editor.Config as config
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.ResizeableText import *

class ComplexShape:

    def __init__( self, anObject, aCanvas, x, y, width, height ):
        self.theCanvas = aCanvas
        self.parentObject = anObject
        self.thePathwayCanvas = anObject.theLayout.getCanvas()
        self.graphUtils = self.parentObject.getGraphUtils()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shapeMap = {}
        self.lastmousex = 0
        self.lastmousey = 0
        self.imageOrigins = {}
        self.shapeLock = {}
        
        self.buttonpressed = False
        self.sumdeltax=0
        self.sumdeltay=0
        self.outlinedrag=False
        
        self.firstdrag=False
        self.dragbefore=False
        
        
        self.outlinedragged=False
        self.objectdragged=False
        self.shapename=None
        self.shift_press=False
        

    def show ( self ):
        canvasRoot = self.parentObject.theCanvas.getRoot()
        self.theRoot = canvasRoot.add(gnomecanvas.CanvasGroup )
        anSD = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST )
        anSD.reCalculate()
        self.shapeDescriptorList = anSD.getDescriptorList()
        
        aDescList = self.__sortByZOrder( self.shapeDescriptorList.values()[:] )

        for aDescriptor in aDescList:
            if aDescriptor[SD_TYPE] == CV_RECT:
                self.createRectangle( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_ELL:
                self.createEllipse( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_TEXT:
                self.createText( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_LINE:
                self.createLine( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_BPATH:
                self.createBpath( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_IMG:
                self.createImage( aDescriptor )
        self.isSelected = False

#-------------------------------------------------------------------------------------------------------------


    def delete( self ):
        
        for aShapeName in self.shapeMap.keys():
            self.shapeMap[ aShapeName ].destroy()
        self.theRoot.destroy()
        self.shapeMap = {}
        self.imageOrigins = {}
        self.shapeLock = {}
        

    def selected( self ):
        self.isSelected = True

    def unselected( self ):
        self.isSelected = False

    def outlineColorChanged( self ):
        # find shapes with outline color
        anRGB = copyValue( self.parentObject.getProperty( OB_OUTLINE_COLOR ) )
        if self.isSelected:
            for i in range(0,3):
                anRGB[i] = 32768 + anRGB[i]
        aColor = self.graphUtils.getGdkColorByRGB( anRGB )

        for aDescriptor in self.shapeDescriptorList.values():
            if aDescriptor[ SD_COLOR ] == SD_OUTLINE:
                if aDescriptor[SD_TYPE] in ( CV_RECT, CV_ELL, CV_BPATH ):
                    self.changeShapeColor( aDescriptor[ SD_NAME ] , aColor)
                elif aDescriptor[SD_TYPE] == CV_LINE:
                    self.changeLineColor( aDescriptor[ SD_NAME ] , aColor )
            elif aDescriptor[ SD_COLOR ] == SD_FILL:
                if aDescriptor[SD_TYPE] in ( CV_RECT, CV_ELL, CV_BPATH ):
                    self.changeShapeColor( aDescriptor[ SD_NAME ] , aColor, True )


    def fillColorChanged( self ):
        # find shapes with outline color
        anRGB = copyValue( self.parentObject.getProperty( OB_FILL_COLOR ) )
        aColor = self.graphUtils.getGdkColorByRGB( anRGB )

        for aDescriptor in self.shapeDescriptorList.values():
            if aDescriptor[ SD_COLOR ] == SD_FILL: 
            
                if aDescriptor[SD_TYPE] == CV_RECT:
                    self.changeShapeColor( aDescriptor[ SD_NAME ] , aColor )
                elif aDescriptor[SD_TYPE] == CV_ELL:
                    self.changeShapeColor( aDescriptor[ SD_NAME ] , aColor )
                elif aDescriptor[SD_TYPE] == CV_LINE:
                    self.changeLineColor( aDescriptor[ SD_NAME ] , aColor )
                elif aDescriptor[SD_TYPE] == CV_BPATH:
                    self.changeShapeColor( aDescriptor[ SD_NAME ] , aColor )


    def labelChanged(self, newLabel):
#       self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).renameLabel(newLabel)
        self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptorList()
        self.renameText(self.shapeDescriptorList['text'])
        self.resize(0,0)


    def move( self, deltax , deltay ):
        self.theRoot.move(deltax,deltay)
        return




    def resize( self, deltawidth, deltaheight ):
        self.width += deltawidth
        self.height += deltaheight
        self.parentObject.getProperty(OB_SHAPEDESCRIPTORLIST).reCalculate()
        for aDescriptor in self.shapeDescriptorList.values():
            if aDescriptor[SD_TYPE] == CV_RECT:
                self.resizeRectangle( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_ELL:
                self.resizeEllipse( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_TEXT:
                self.resizeText( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_LINE:
                self.resizeLine( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_BPATH:
                self.resizeBpath( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_IMG:
                self.resizeImage( aDescriptor )
        
        
    def resizeBpath( self, aDescriptor ):
        pathDef =  aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aBpath = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        self.setOutlineWidth( aDescriptor, aBpath )
        newPathDef = []
        for anArtPath in pathDef:
            newArtPath=[ anArtPath[0]]
            for i in range(0,(len(anArtPath) - 1 )/2):
                x,y = aBpath.w2i(anArtPath[i*2+1], anArtPath[i*2+2] )
                newArtPath.extend( [x,y] )
            newArtPath = tuple(newArtPath)
            newPathDef.append( newArtPath )
        aBpath.set_bpath(gnomecanvas.path_def_new( newPathDef ) )
        

    def createBpath( self, aDescriptor ):
        pathDef =  aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        outlineColor = self.getOutlineColor( )
        aBpath = self.theRoot.add( gnomecanvas.CanvasBpath,  outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
        aBpath.set_bpath(gnomecanvas.path_def_new( pathDef ) )
        self.setOutlineWidth( aDescriptor, aBpath )
        self.addHandlers( aBpath, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aBpath
        


    def setOutlineWidth( self, aDescriptor, aShape ):
        outlineRatio = aDescriptor[SD_SPECIFIC][SPEC_WIDTH_RATIO]
        outlineWidth = self.parentObject.getProperty( OB_OUTLINE_WIDTH )
        outlineWidth *= outlineRatio
        aShape.set_property( "width-units", outlineWidth )


    def createRectangle( self, aDescriptor ):
        ( X1, Y1, X2, Y2 ) =  aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        outlineColor = self.getOutlineColor( )
        aRect = self.theRoot.add( gnomecanvas.CanvasRect, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
        self.setOutlineWidth( aDescriptor, aRect )
        self.addHandlers( aRect, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aRect


    def resizeRectangle( self, aDescriptor ):
        ( X1, Y1, X2, Y2 ) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        (X1, Y1) = aShape.w2i( X1, Y1 )
        (X2, Y2) = aShape.w2i( X2, Y2 )

        aShape.set_property( 'x1', X1 )
        aShape.set_property( 'y1', Y1 )
        aShape.set_property( 'x2', X2 )
        aShape.set_property( 'y2', Y2 )


    def changeShapeColor ( self, shapeName, aColor, flag = False ):
        aShape = self.shapeMap[ shapeName  ]
        if flag :
            aShape.set_property('outline_color_gdk', aColor )
        else:
            aShape.set_property('fill_color_gdk', aColor )              



    def createEllipse( self, aDescriptor ):
        ( X1, Y1, X2, Y2 ) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        outlineColor = self.getOutlineColor( )
        anEllipse = self.theRoot.add( gnomecanvas.CanvasEllipse, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
        self.setOutlineWidth( aDescriptor, anEllipse )

        self.addHandlers( anEllipse, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = anEllipse


    def resizeEllipse( self, aDescriptor ):
        ( X1, Y1, X2, Y2 ) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        (X1, Y1) = aShape.w2i( X1, Y1 )
        (X2, Y2) = aShape.w2i( X2, Y2 )
        aShape.set_property( 'x1', X1 )
        aShape.set_property( 'y1', Y1 )
        aShape.set_property( 'x2', X2 )
        aShape.set_property( 'y2', Y2 )


    def createLine( self, aDescriptor ):
        lineSpec = aDescriptor[SD_SPECIFIC]
        ( X1, Y1, X2, Y2 ) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        aLine = self.theRoot.add( gnomecanvas.CanvasLine,points=[X1,Y1,X2,Y2], width_units=lineSpec[ LINE_WIDTH ], fill_color_gdk = aGdkColor )
        self.addHandlers( aLine, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aLine


    def resizeLine( self, aDescriptor ):
        ( X1, Y1, X2, Y2 ) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        (X1, Y1) = aShape.w2i( X1, Y1 )
        (X2, Y2) = aShape.w2i( X2, Y2 )
        aShape.set_property( 'points', [X1,Y1,X2,Y2] )


    def changeLineColor ( self, shapeName, aColor ):
        aShape = self.shapeMap[  shapeName ] 
        aclr = aShape.get_property('fill_color_gdk')
        aShape.set_property('fill_color_gdk', aColor )
        aclr = aShape.get_property('fill_color_gdk')


####################################################################################################
    def createText( self, aDescriptor ):
        textSpec = aDescriptor[SD_SPECIFIC]
        (X1, Y1) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        #aText = self.theRoot.add( gnomecanvas.CanvasText,x=X1,y=Y1, fill_color_gdk = aGdkColor, text = textSpec[SPEC_LABEL], anchor #= gtk.ANCHOR_NW )
        #parentID=self.parentObject.getProperty(OB_FULLID)
        aText = ResizeableText( self.theRoot, self.theCanvas, X1, Y1, aGdkColor, textSpec[SPEC_LABEL], gtk.ANCHOR_NW )
        self.addHandlers( aText, aDescriptor[ SD_NAME ] )
        #aText.addHandlers(aDescriptor[ SD_NAME ])
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aText

    def resizeText( self, aDescriptor ):
        #by default text cannot be resized, it defines size 
        (x1, y1) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        (x1, y1) = aShape.w2i( x1, y1 )
        aShape.set_property( 'x', x1 )
        aShape.set_property( 'y', y1 )

    def renameText( self, aDescriptor ):
        textSpec = aDescriptor[SD_SPECIFIC]
        text=textSpec[SPEC_LABEL]
        self.shapeMap[ aDescriptor[ SD_NAME ] ].set_property('text', text )

    def createImage( self, aDescriptor ):
        imgSpec = aDescriptor[SD_SPECIFIC]
        filename = config.SHAPE_PLUGIN_PATH +  imgSpec[IMG_FILENAME]
        anImage = gtk.Image( )
        anImage.set_from_file( filename )
        aPixbuf = anImage.get_property("pixbuf")
        aWidth = aPixbuf.get_property("width" )
        aHeight = aPixbuf.get_property( "height" )
        eventBox = gtk.EventBox()
        eventBox.add( anImage )
        eventBox.set_events( gtk.gdk.POINTER_MOTION_MASK| gtk.gdk.BUTTON_PRESS_MASK| gtk.gdk.BUTTON_RELEASE_MASK |gtk.gdk.ENTER_NOTIFY_MASK )
        aName = aDescriptor [SD_NAME]
        eventBox.show_all()
        eventBox.connect( "motion-notify-event", self.img_event, aName )
        eventBox.connect( "button-press-event", self.img_event, aName )
        eventBox.connect( "button-release-event", self.img_event, aName )
        eventBox.connect("enter-notify-event", self.enter_img )
        x1,y1 = imgSpec[SPEC_POINTS]

        imgShape = self.theRoot.add( gnomecanvas.CanvasWidget, x=x1, y=y1, width = aWidth, height = aHeight, widget = eventBox )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = imgShape        
        self.imageOrigins[ aDescriptor[ SD_NAME ] ] = [ x1, y1 ]
        self.shapeLock [ aDescriptor[ SD_NAME ] ] = False


    def resizeImage( self, aDescriptor ):
        (x1, y1) = aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        (x1, y1) = aShape.w2i( x1, y1 )
        aShape.set_property( 'x', x1 )
        aShape.set_property( 'y', y1 )
        

    def img_event( self, *args ):
        item = args[0]
        event = ArtifitialEvent( args[1] )
#        groupx, groupy = self.theRoot.get_property("x") , self.theRoot.get_property("y")
#        allo = item.get_allocation()
#        (a,b,c,d,offsx, offsy)= self.theRoot.i2c_affine( (0,0,0,0,0,0))

#        event.x, event.y = groupx - offsx + event.x+allo.x, groupy - offsy + event.y+allo.y
        shapeName = args[2]
        shape = self.shapeMap[ shapeName ]
        origx, origy = self.imageOrigins[ shapeName ]
        zerox, zeroy = shape.w2i( 0,0 )

        relx, rely = shape.w2i( event.x, event.y )
        pixx, pixxy, flags =self.theCanvas.getCanvas().bin_window.get_pointer( )
        worldx, worldy = self.theCanvas.getCanvas().window_to_world( pixx,pixxy)
        event.x =  worldx
        event.y =  worldy


        self.rect_event( item, event, shapeName )
        
    def enter_img( self, *args ):
        self.thePathwayCanvas.beyondcanvas = False

    def getGdkColor( self, aDescriptor ):
        aColorType = aDescriptor[ SD_COLOR ]
        if aColorType == SD_FILL:
            queryProp = OB_FILL_COLOR
        elif aColorType == SD_OUTLINE:
            queryProp = OB_OUTLINE_COLOR
        elif aColorType == SD_TEXT:
            queryProp = OB_TEXT_COLOR
        
        anRGBColor = self.parentObject.getProperty( queryProp )
        
        return self.graphUtils.getGdkColorByRGB( anRGBColor )

    def getOutlineColor( self ):
        anRGBColor = self.parentObject.getProperty( OB_OUTLINE_COLOR )
        return self.graphUtils.getGdkColorByRGB( anRGBColor )


    def __sortByZOrder ( self, desclist ):
        fn = lambda x, y: ( x[SD_Z] < y[SD_Z] ) - ( y[SD_Z] < x[SD_Z] )
        desclist.sort(fn)
        return desclist
            


    def leftClick( self, shapeName, x, y):
        # usually select
        
        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_SYSTEM_CANVAS:
            self.parentObject.addItem( x, y )
        else:
            self.parentObject.doSelect()


        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] in [ SD_FILL, SD_RING ]:
            self.changeCursor( shapeName, x, y, True )


    def rightClick ( self, shapeName, x, y, anEvent, shift=False ):
        # usually show menu
        if not self.parentObject.isSelected:
            self.parentObject.doSelect( shift )
        self.parentObject.showMenu(anEvent,x, y)

    def SHIFT_leftClick ( self, shapeName, x, y):
      
        self.parentObject.doSelect( True )

        
    def mouseDrag( self, shapeName, deltax, deltay, origx, origy ):
        # decide whether resize or move or draw arrow
        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_OUTLINE:
            if not self.firstdrag and not self.dragbefore:
                self.outlinedragged=True
                self.firstdrag=True
            self.parentObject.outlineDragged( deltax, deltay, origx, origy )
        elif self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_RING:
            self.parentObject.ringDragged( shapeName, deltax, deltay )
        elif self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_FILL:
            if not self.firstdrag and not self.dragbefore:
                self.firstdrag=True
                self.objectdragged=True
                self.orgx=origx
                self.orgy=origy
                self.shapename=shapeName
            self.parentObject.objectDragged( deltax, deltay )
            self.setCursor(CU_MOVE)

    def  doubleClick( self, shapeName ):
        self.parentObject.popupEditor()


    def getShapeDescriptor( self, shapeName ):
        return self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptor( shapeName )

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def addHandlers( self, canvasObject, aName ):
        #canvasObject.connect('event', self.rect_event, aName )
        canvasObject.connect('event', self.rect_event, aName )

    def releaseButton( self, shapeName, x, y ):
        self.changeCursor( shapeName, x, y, False )
        self.parentObject.buttonReleased()

    def mouseEntered( self, shapeName, x, y ):
        self.changeCursor( shapeName, x, y )

    def changeCursor( self, shapeName, x, y, buttonpressed  = False):
        aFunction = self.getShapeDescriptor(shapeName)[SD_FUNCTION]
        #self.parentObject.__class__.__name__ ,'H'
        aCursorType = self.parentObject.getCursorType( aFunction, x, y , buttonpressed)
        self.theCanvas.setCursor( aCursorType )

    def setCursor( self, aCursorType):
        self.theCanvas.setCursor( aCursorType )
    
    def getFirstDrag(self):
        return self.firstdrag
    
    def setFirstDrag(self,aValue):
        self.firstdrag=aValue

    def getDragBefore(self):
        return self.dragbefore
    
    def setDragBefore(self,aValue):
        self.dragbefore=aValue

    def getIsButtonPressed(self):
        return self.buttonpressed

    def getOutlineDragged(self):
        return self.outlinedragged

    def getObjectDragged(self):
        return self.objectdragged

    def rect_event( self, *args ):

        event = args[1]
        item = args[0]
        shapeName = args[2]
        
        if event.type == gtk.gdk.BUTTON_PRESS:
            if event.button == 1:
                self.lastmousex = event.x
                self.lastmousey = event.y
                self.buttonpressed = True

                if event.state&gtk.gdk.SHIFT_MASK == gtk.gdk.SHIFT_MASK:
                    self.shift_press = True
                    self.SHIFT_leftClick( shapeName, event.x, event.y)

                else: 
                    self.shift_press = False
                    self.leftClick( shapeName, event.x, event.y )

            elif event.button == 3:
                if event.state&gtk.gdk.SHIFT_MASK == gtk.gdk.SHIFT_MASK:
                    self.shift_press = True
                else: 
                    self.shift_press = False

                self.rightClick(shapeName, event.x, event.y, event, self.shift_press )

        elif event.type == gtk.gdk.BUTTON_RELEASE:
            if event.button == 1:
                self.buttonpressed = False
                if self.dragbefore:
                    self.dragbefore=False
                if self.objectdragged:  
                    self.parentObject.objectDragged( 0,0 )
                    self.objectdragged=False
                if self.outlinedragged:
                    self.parentObject.outlineDragged( 0, 0, 0, 0 )
                    self.outlinedragged=False
                self.releaseButton(shapeName, event.x, event.y )

        elif event.type == gtk.gdk.MOTION_NOTIFY:
            if not self.buttonpressed:
                return
            if (event.state&gtk.gdk.BUTTON1_MASK)==0:
                return
            oldx = self.lastmousex
            oldy = self.lastmousey
            deltax = event.x - oldx
            deltay = event.y - oldy
            self.lastmousex = event.x
            self.lastmousey = event.y
            if not self.parentObject.theLayout.getCanvas().getRecentScroll():
                # if there was a scroll, event.x, event.y gets a stupid value
                self.mouseDrag( shapeName, deltax, deltay, oldx, oldy )
            
        elif event.type == gtk.gdk._2BUTTON_PRESS:
            if event.button == 1:
                self.doubleClick( shapeName )

        elif event.type == gtk.gdk.ENTER_NOTIFY:
            self.mouseEntered( shapeName, event.x, event.y )


        
class ArtifitialEvent:
    def __init__( self, anEvent ):
        self.type = anEvent.type
        self.x = anEvent.x
        self.y = anEvent.y
        self.state = anEvent.state
        self.time = anEvent.time
        if anEvent.type not in [ gtk.gdk.ENTER_NOTIFY, gtk.gdk.MOTION_NOTIFY ]:
            self.button = anEvent.button
            
