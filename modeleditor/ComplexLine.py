import gnome.canvas
from Constants import *
from Utils import *
from ResizeableText import *

class ComplexLine:

    def __init__( self, anObject, aCanvas ):
        self.theCanvas = aCanvas
        self.parentObject = anObject
        self.graphUtils = self.parentObject.getGraphUtils()
        self.shapeMap = {}
        self.lastmousex = 0
        self.lastmousey = 0
        self.buttonpressed = False
        self.firstdrag=False



    def show ( self ):
        self.theRoot = self.parentObject.theCanvas.getRoot()
        self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptorList()
        self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).reCalculate()
        self.__sortByZOrder( self.shapeDescriptorList )
        self.isSelected = False
        for aKey in self.shapeDescriptorList.keys():
            aDescriptor = self.shapeDescriptorList[aKey]
            if aDescriptor[SD_TYPE] == CV_TEXT:
                self.createText( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_LINE:
                self.createLine( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_BPATH:
                self.createBpath( aDescriptor )
        self.isSelected = False
    

    def repaint ( self ):
        self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).reCalculate()
        self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptorList()
        self.__sortByZOrder( self.shapeDescriptorList )
        for aKey in self.shapeDescriptorList.keys():
            aDescriptor = self.shapeDescriptorList[aKey]
            if aDescriptor[SD_TYPE] == CV_TEXT:
                self.redrawText( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_LINE:
                self.redrawLine( aDescriptor )
            elif aDescriptor[SD_TYPE] == CV_BPATH:
                self.redrawBpath( aDescriptor )
        

    def reName( self ):
        self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptorList()
        self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).renameLabel( self.parentObject.getProperty( CO_NAME ) )
        aDescriptor = self.shapeDescriptorList["textbox"]
        self.renameText( aDescriptor )
        

    def delete( self ):
        for aShapeName in self.shapeMap.keys():
            self.shapeMap[ aShapeName ].destroy()


    def selected( self ):
        self.isSelected = True


    def unselected( self ):
        self.isSelected = False


    def outlineColorChanged( self ):
        
        self.fillColorChanged()

    def fillColorChanged( self ):
        # find shapes with outline color
        anRGB = copyValue( self.parentObject.getProperty( OB_FILL_COLOR ) )
        if self.isSelected:
            for i in range(0,3):
                anRGB[i] = 32768 + anRGB[i]
        
        for aKey in self.shapeDescriptorList.keys():
            aDescriptor = self.shapeDescriptorList[aKey]
            if aDescriptor[ SD_COLOR ] == SD_FILL:
                aColor = self.graphUtils.getGdkColorByRGB( anRGB )
                if aDescriptor[SD_TYPE] == CV_LINE:
                    self.changeLineColor( aDescriptor[ SD_NAME ] , aColor )

    
    def createBpath(self, aDescriptor):
        aSpecific= aDescriptor[SD_SPECIFIC]
        # get pathdef
        pathdef= aSpecific[BPATH_PATHDEF]
        pd=gnome.canvas.path_def_new(pathdef)

        aGdkColor = self.getGdkColor( aDescriptor )
        bpath=self.theRoot.add(gnome.canvas.CanvasBpath, width_units=3, 
outline_color_gdk = aGdkColor )
        bpath.set_bpath(pd)
    
        self.addHandlers( bpath, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = bpath
        
    def createLine( self, aDescriptor ):
        lineSpec = aDescriptor[SD_SPECIFIC]
        
        ( X1, X2, Y1, Y2 ) = [lineSpec[0], lineSpec[2], lineSpec[1], lineSpec[3] ]
        
        aGdkColor = self.getGdkColor( aDescriptor )
        firstArrow = lineSpec[4]
        secondArrow = lineSpec[5]

        aLine = self.theRoot.add( gnome.canvas.CanvasLine,points=[X1,Y1,X2,Y2], width_units=lineSpec[ 6 ], fill_color_gdk = aGdkColor, first_arrowhead = firstArrow, last_arrowhead = secondArrow,arrow_shape_a=5, arrow_shape_b=5, arrow_shape_c=3 )
        self.addHandlers( aLine, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aLine



    def changeLineColor ( self, shapeName, aColor ):
        aShape = self.shapeMap[  shapeName ] 
        aclr = aShape.get_property('fill_color_gdk')
        aShape.set_property('fill_color_gdk', aColor )
        aclr = aShape.get_property('fill_color_gdk')


    def createText( self, aDescriptor ):
        textSpec = aDescriptor[SD_SPECIFIC]
        (X1, Y1) = ( textSpec[TEXT_ABSX], textSpec[TEXT_ABSY] )
        aGdkColor = self.getGdkColor( aDescriptor )
        aText = ResizeableText( self.theRoot, self.theCanvas, X1, Y1, aGdkColor, textSpec[TEXT_TEXT], gtk.ANCHOR_NW )
        self.addHandlers( aText, aDescriptor[ SD_NAME ] )
        self.shapeMap[ aDescriptor[ SD_NAME ] ] = aText


    def redrawLine( self, aDescriptor ):
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        x1 = aSpecific[0]
        y1 = aSpecific[1]
        x2 = aSpecific[2]
        y2 = aSpecific[3]
        hasFirstArrow = aSpecific[4]
        hasLastArrow = aSpecific[5]
        aShape.set_property( 'points', (x1, y1, x2, y2) )
        aShape.set_property('first_arrowhead', hasFirstArrow )
        aShape.set_property('last_arrowhead', hasLastArrow )

    def redrawBpath( self, aDescriptor ):
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        pathdef = aDescriptor[ SD_SPECIFIC ][BPATH_PATHDEF]
        pd=gnome.canvas.path_def_new(pathdef)
        aShape.set_bpath(pd)
        

    def redrawText( self, aDescriptor ):
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        x = aSpecific[TEXT_ABSX]
        y = aSpecific[TEXT_ABSY]
        aShape.set_property( 'x', x )
        aShape.set_property( 'y', y )
        


    def renameText (self, aDescriptor ):
        aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        label = aSpecific[ TEXT_TEXT ]
        aShape.set_property( 'text', label )


    def getGdkColor( self, aDescriptor ):
        aColorType = aDescriptor[ SD_COLOR ]
        if aColorType == SD_FILL:
            queryProp = OB_FILL_COLOR
        elif aColorType == CV_TEXT:
            queryProp = OB_TEXT_COLOR

        anRGBColor = self.parentObject.getProperty( queryProp )

        return self.graphUtils.getGdkColorByRGB( anRGBColor )



    def __sortByZOrder ( self, desclist ):
        keys = desclist.keys()
        fn = lambda x, y: ( x[SD_Z] < y[SD_Z] ) - ( y[SD_Z] < x[SD_Z] )

        keys.sort(fn)


    def leftClick( self, shapeName, x, y ):
        # usually select
        self.parentObject.doSelect()
        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_ARROWHEAD:
            self.changeCursor( shapeName, x, y, True )


    def rightClick ( self, shapeName, x, y, anEvent ):
        # usually show menu
        self.parentObject.doSelect()
        self.parentObject.showMenu( anEvent)

    def getFirstDrag(self):
        return self.firstdrag
    
    def setFirstDrag(self,aValue):
        self.firstdrag=aValue

    def mouseDrag( self, shapeName, deltax, deltay, origx, origy ):
        # decide whether resize or move or draw arrow
        if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_MOVINGLINE:
            print "movingline dragged"
        elif self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_ARROWHEAD:
            if not self.firstdrag:
                self.firstdrag=True
            self.parentObject.arrowheadDragged( shapeName,deltax, deltay, origx, origy)

    
        
    def checkConnection( self ):
        self.parentObject.checkConnection()


    def  doubleClick( self, shapeName ):
        self.parentObject.popupEditor()


    def getShapeDescriptor( self, shapeName ):
        return self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptor( shapeName )


    def addHandlers( self, canvasObject, aName ):
        canvasObject.connect('event', self.rect_event, aName )


    def releaseButton( self, shapeName, x, y ):
        self.changeCursor( shapeName, x, y, False )
        self.parentObject.mouseReleased( shapeName,x, y)

    def mouseEntered( self, shapeName, x, y ):
        self.changeCursor( shapeName, x, y )


    def changeCursor( self, shapeName, x, y, buttonpressed  = False):
        aFunction = self.getShapeDescriptor(shapeName)[SD_FUNCTION]
        aCursorType = self.parentObject.getCursorType( aFunction, x, y , buttonpressed)
        self.theCanvas.setCursor( aCursorType )


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
                self.releaseButton(shapeName, event.x, event.y )
                

        elif event.type == gtk.gdk.MOTION_NOTIFY:
            self.buttonpressed=(event.state&gtk.gdk.BUTTON1_MASK)>0
            if not self.buttonpressed:
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

    
