import gnome.canvas
from Constants import *
from Utils import *
from ResizeableText import *


class ComplexShape:

    def __init__( self, anObject, aCanvas, x, y, width, height ):
        self.theCanvas = aCanvas
        self.parentObject = anObject
        self.graphUtils = self.parentObject.getGraphUtils()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.shapeMap = {}
        self.lastmousex = 0
        self.lastmousey = 0
        
        self.buttonpressed = False
        self.sumdeltax=0
        self.sumdeltay=0
        self.outlinedrag=False
        
        self.firstdrag=False
        self.dragbefore=False
        
        
        self.outlinedragged=False
        self.objectdragged=False
        self.shapename=None
        

    def show ( self ):
        canvasRoot = self.parentObject.theCanvas.getRoot()
        self.theRoot = canvasRoot.add(gnome.canvas.CanvasGroup )
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
        self.isSelected = False

#-------------------------------------------------------------------------------------------------------------


    def delete( self ):
        
        for aShapeName in self.shapeMap.keys():
            self.shapeMap[ aShapeName ].destroy()
        
        

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
        self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).renameLabel(newLabel)
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
        aBpath.set_bpath(gnome.canvas.path_def_new( newPathDef ) )
        

    def createBpath( self, aDescriptor ):
        pathDef =  aDescriptor[SD_SPECIFIC][SPEC_POINTS]
        aGdkColor = self.getGdkColor( aDescriptor )
        outlineColor = self.getOutlineColor( )
        aBpath = self.theRoot.add( gnome.canvas.CanvasBpath,  outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
        aBpath.set_bpath(gnome.canvas.path_def_new( pathDef ) )
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
        aRect = self.theRoot.add( gnome.canvas.CanvasRect, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
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
        anEllipse = self.theRoot.add( gnome.canvas.CanvasEllipse, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = outlineColor, fill_color_gdk = aGdkColor )
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
        aLine = self.theRoot.add( gnome.canvas.CanvasLine,points=[X1,Y1,X2,Y2], width_units=lineSpec[ LINE_WIDTH ], fill_color_gdk = aGdkColor )
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
        #aText = self.theRoot.add( gnome.canvas.CanvasText,x=X1,y=Y1, fill_color_gdk = aGdkColor, text = textSpec[SPEC_LABEL], anchor #= gtk.ANCHOR_NW )
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
            


    def leftClick( self, shapeName, x, y ):
        # usually select
        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_SYSTEM_CANVAS:
            self.parentObject.addItem( x, y )
        else:
            self.parentObject.doSelect()


        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] in [ SD_FILL, SD_RING ]:
            self.changeCursor( shapeName, x, y, True )


    def rightClick ( self, shapeName, x, y, anEvent ):
        # usually show menu
        self.parentObject.doSelect()
        self.parentObject.showMenu(anEvent,x, y)

        
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

                self.leftClick( shapeName, event.x, event.y )
            elif event.button == 3:
                self.rightClick(shapeName, event.x, event.y, event )

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
            self.mouseDrag( shapeName, deltax, deltay, oldx, oldy )
            
        elif event.type == gtk.gdk._2BUTTON_PRESS:
            if event.button == 1:
                self.doubleClick( shapeName )

        elif event.type == gtk.gdk.ENTER_NOTIFY:
            self.mouseEntered( shapeName, event.x, event.y )


        
