from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *


#from System import *




class SystemObject(EditorObject):


    def __init__( self, aLayout, objectID, aFullID,  x, y , parentSystem ):
        EditorObject.__init__( self, aLayout, objectID, x, y, parentSystem )
        self.thePropertyMap[ OB_HASFULLID ] = True
        self.thePropertyMap [ OB_FULLID ] = aFullID
        self.theObjectMap = {}
        self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
        self.thePropertyMap[ OB_TYPE ] = OB_TYPE_SYSTEM
        self.thePropertyMap [ OB_DIMENSION_X ]=SYS_MINWIDTH
        self.thePropertyMap [ OB_DIMENSION_Y ]=SYS_MINHEIGHT
        self.thePropertyMap [ OB_LABEL ]=aFullID
        self.thePropertyMap [ OB_MINLABEL ]=SYS_MINLABEL
        self.theLabel = self.thePropertyMap [ OB_LABEL ]

        self.theMaxShiftPosx=0;self.theMaxShiftPosy=0
        self.theMaxShiftNegx=0;self.theMaxShiftNegy=0
        self.theorgdir=0
        self.accshiftx=0;self.accshifty=0
        self.prect1=None;self.prect2=None;self.rectdotx=0;self.rectdoty=0
        
        self.thex1org=0
        self.they1org=0
        self.thex2org=0
        self.they2org=0
        self.theduporg=0
        self.theddownorg=0
        self.thedleftorg=0
        self.thedrightorg=0
        
        aSystemSD=EditorObject.getShapeDescriptor(self, self.getProperty( OB_SHAPE_TYPE ) )
        
        reqWidth = aSystemSD.getRequiredWidth()
        reqHeight = aSystemSD.getRequiredHeight()
        
        if parentSystem.__class__.__name__ == 'Layout':
            layoutDims = self.theLayout.getProperty( LO_SCROLL_REGION )
            self.thePropertyMap [ OB_DIMENSION_X ] = layoutDims[2] - layoutDims[0]-1
            self.thePropertyMap [ OB_DIMENSION_Y ] = layoutDims[3]- layoutDims[1]-1
        else:
            lblWidth=reqWidth
            x=self.getProperty(OB_POS_X)
            y=self.getProperty(OB_POS_Y)
            x2=x+self.getProperty(OB_DIMENSION_X)
            y2=y+self.getProperty(OB_DIMENSION_Y)
            px2=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)
            py2=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)
            rpar=n.array([0,0,px2,py2])
            rn=self.createRnOut()
            availspace=self.parentSystem.getAvailSpace(x,y,x2,y2,rn)
            #checkLabel
            if lblWidth>SYS_MINWIDTH:
                newLabel=self.truncateLabel(self.theLabel,lblWidth,SYS_MINWIDTH)
                self.thePropertyMap [OB_LABEL]=newLabel
                aSystemSD.renameLabel (newLabel)
                
            lblWidth=aSystemSD.getRequiredWidth()
            
            largest=max(availspace/2,lblWidth)
            self.thePropertyMap [ OB_DIMENSION_X ]=largest
            if largest==availspace/2:
                diff=self.getProperty(OB_DIMENSION_X)-(x2-x)
                self.thePropertyMap [ OB_DIMENSION_Y ]=y2-y+diff
        self.setShapeDescriptor( aSystemSD )

        self.thePropertyMap[ SY_INSIDE_DIMENSION_X  ] = aSystemSD.getInsideWidth()
        self.thePropertyMap[ SY_INSIDE_DIMENSION_Y  ] = aSystemSD.getInsideHeight()


        

        self.cursorMap={ DIRECTION_TOP_LEFT:CU_RESIZE_TOP_LEFT, DIRECTION_UP:CU_RESIZE_TOP, 
                 DIRECTION_TOP_RIGHT:CU_RESIZE_TOP_RIGHT, DIRECTION_RIGHT:CU_RESIZE_RIGHT,
                     DIRECTION_BOTTOM_RIGHT:CU_RESIZE_BOTTOM_RIGHT, DIRECTION_DOWN:CU_RESIZE_BOTTOM, 
                 DIRECTION_BOTTOM_LEFT:CU_RESIZE_BOTTOM_LEFT, DIRECTION_LEFT:CU_RESIZE_LEFT}

        self.dragMap={DIRECTION_LEFT:n.array([[1,0],[0,0],[0,0],[0,0]]), 
                 DIRECTION_RIGHT:n.array([[0,0],[0,0],[1,0],[0,0]]),
                 DIRECTION_UP:n.array([[0,0],[0,1],[0,0],[0,0]]), 
                 DIRECTION_DOWN:n.array([[0,0],[0,0],[0,0],[0,1]]),
                 DIRECTION_BOTTOM_RIGHT:n.array([[0,0],[0,0],[1,0],[0,1]]), 
                 DIRECTION_BOTTOM_LEFT:n.array([[-1,0],[0,0],[0,0],[0,1]]),
                 DIRECTION_TOP_RIGHT:n.array([[0,0],[0,-1],[1,0],[0,0]]), 
                 DIRECTION_TOP_LEFT:n.array([[-1,0],[0,-1],[0,0],[0,0]])}


        #self.UDLRMap ={'U':0,'D':1,'L':2,'R':3}
        self.UDLRMap ={'L':0,'U':1,'R':2,'D':3}

    def destroy( self ):
        for anObjectID in self.theObjectMap.keys()[:]:
            self.theLayout.deleteObject( anObjectID )
        EditorObject.destroy( self )

    def move( self, deltax, deltay ):
        EditorObject.move( self, deltax,deltay)
        for anObjectID in self.theObjectMap.keys():
            self.theObjectMap[ anObjectID ].parentMoved( deltax, deltay )

    def registerObject( self, anObject ):
        self.theObjectMap[anObject.getID()] = anObject

    def unregisterObject ( self, anObjectID ):
        self.theObjectMap.__delitem__( anObjectID )

    def parentMoved( self, deltax, deltay ):
        EditorObject.parentMoved( self, deltax, deltay )
        for anID in self.theObjectMap.keys():
            self.theLayout.getObject( anID ).parentMoved( deltax, deltay )
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\

    def pasteObject(self):
        (offsetx, offsety ) = self.getAbsolutePosition()
        x = self.newObjectPosX - (self.theSD.insideX + offsetx )
        y = self.newObjectPosY - ( self.theSD.insideY + offsety )
        aBuffer = self.getModelEditor().getCopyBuffer()
        #aType = aBuffer.getProperty(OB_TYPE)
        x2 = x+aBuffer.getProperty(OB_DIMENSION_X)
        y2 = y+aBuffer.getProperty(OB_DIMENSION_Y)
        px2=self.getProperty(SY_INSIDE_DIMENSION_X)
        py2=self.getProperty(SY_INSIDE_DIMENSION_Y)
        rpar=n.array([0,0,px2,py2])
        rn=self.createRnIn()
        availspace=self.getAvailSpace(x,y,x2,y2,rn)
        if availspace>0:
            if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                aCommand =PasteObject( self.theLayout, aBuffer,x,y, self.theID )
                self.theLayout.passCommand( [aCommand] )
            else:
                # change cursor
                self.theShape.setCursor(CU_CROSS)
        else:
            self.theShape.setCursor(CU_CROSS)       


    def canPaste(self):
        aBuffer = self.getModelEditor().getCopyBuffer()
        aType = aBuffer.getType()
        aParentFullID = self.getProperty(OB_FULLID)
        aSystemPath = convertSysIDToSysPath( aParentFullID )

        if aType == 'SystemObjectBuffer':
            self.setExistObjectFullIDList()
            return self.canPasteOneSystemBuffer( aBuffer, aSystemPath )
        elif aType == "MultiObjectBuffer":
            self.setExistObjectFullIDList()
            for aSystemBufferName in aBuffer.getSystemObjectListBuffer().getObjectBufferList():
                aSystemBuffer = aBuffer.getSystemObjectListBuffer().getObjectBuffer( aSystemBufferName )
                if not self.canPasteOneSystemBuffer( aSystemBuffer, aSystemPath ):
                    return False
                
#            for aBufferName in aBuffer.getObjectListBuffer().getObjectBufferList():
#                anObjectBuffer = aBuffer.getObjectListBuffer().getObjectBuffer( aBufferName )
#                if not self.canPasteOneSystemBuffer( anObjectBuffer, aParentFullID ):
#                    return False
            return True
        else:
            return True


    def canPasteOneSystemBuffer( self, aBuffer, aSystemPath ):
        anObjName = aBuffer.getProperty( OB_FULLID ).split(':')[2]
        aFullID = ":".join( [ME_SYSTEM_TYPE, aSystemPath, anObjName ] )
        if not self.theModelEditor.getModel().isEntityExist( aFullID):
            return True
        else:
            return False
        

    def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
        #first do a resize then a move
        # FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
        # resize must be sum of deltas
        self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
        self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown 
        self.thePropertyMap[ SY_INSIDE_DIMENSION_X ] += deltaleft + deltaright
        self.thePropertyMap[ SY_INSIDE_DIMENSION_Y ] += deltaup + deltadown 
        if self.theShape!= None:
            self.theShape.resize( deltaleft + deltaright, deltaup + deltadown )
        if deltaleft!= 0 or deltaup !=0:
            self.move( -deltaleft, -deltaup )
        
    def setProperty(self, aPropertyName, aPropertyValue):
       
        if  self.theCanvas !=None:
            if aPropertyName == OB_DIMENSION_X :
                oldx = self.thePropertyMap[ OB_DIMENSION_X ]
                deltaright = aPropertyValue - oldx
                self.resize( 0,0,0,deltaright )
                return
            if aPropertyName == OB_DIMENSION_Y :
                oldy = self.thePropertyMap[ OB_DIMENSION_Y ]
                deltadown = aPropertyValue - oldy
                self.resize( 0,deltadown,0,0 )
                return
            #if aPropertyName == OB_LABEL:
            #    return
        EditorObject.setProperty(self, aPropertyName, aPropertyValue)
            
    def estLabelWidth(self,newLabel):
        height,width=self.getGraphUtils().getTextDimensions(newLabel)
        return width+16

    def labelChanged( self,aPropertyValue ):
        newLabel=aPropertyValue
        #totalWidth,limit=self.getLabelParam()
        #if totalWidth>limit:
        #   newLabel=self.truncateLabel(newLabel,totalWidth,limit)
        #   self.thePropertyMap[OB_LABEL]=newLabel
        
        self.theShape.labelChanged(self.getProperty(OB_LABEL)) 
        
    def getEmptyPosition( self ):
        return ( 50,50 )


    def show( self ):
        #render to canvas
        EditorObject.show( self )

    def addItem( self, absx,absy ):
        (offsetx, offsety ) = self.getAbsolutePosition()
        x = absx - (self.theSD.insideX + offsetx )
        y = absy - ( self.theSD.insideY + offsety )
        aSysPath = convertSysIDToSysPath( self.getProperty( OB_FULLID ) )
        aCommand = None
        buttonPressed = self.theLayout.getPaletteButton()
        px2=self.getProperty(SY_INSIDE_DIMENSION_X)
        py2=self.getProperty(SY_INSIDE_DIMENSION_Y)
        rpar=n.array([0,0,px2,py2])

        if  buttonPressed == PE_SYSTEM:
            # create command
            aName = self.getModelEditor().getUniqueEntityName ( ME_SYSTEM_TYPE, aSysPath )
            aFullID = ':'.join( [ME_SYSTEM_TYPE, aSysPath, aName] )
            objectID = self.theLayout.getUniqueObjectID( OB_TYPE_SYSTEM )

            # check boundaries
            rn=self.createRnAddSystem()
            minreqx, minreqy = self.getMinDims( ME_SYSTEM_TYPE, aFullID )
            x2=x+max( SYS_MINWIDTH, minreqx )
            y2=y+max( SYS_MINHEIGHT, minreqy )

            availspace=self.getAvailSpace(x,y,x2,y2,rn)
            if availspace>0:
                if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                    
                    aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_SYSTEM, aFullID, x, y, self )
                else:
                    self.theShape.setCursor(CU_CROSS)
            else:
                # change cursor
                self.theShape.setCursor(CU_CROSS)


        elif buttonPressed == PE_PROCESS:
            # create command
            aName = self.getModelEditor().getUniqueEntityName ( ME_PROCESS_TYPE, aSysPath )
            aFullID = ':'.join( [ME_PROCESS_TYPE, aSysPath, aName] )
            objectID = self.theLayout.getUniqueObjectID( OB_TYPE_PROCESS )
            minreqx, minreqy = self.getMinDims( ME_PROCESS_TYPE, aFullID.split(":")[2] )
            x2=x+max( PRO_MINWIDTH, minreqx )
            y2=y+max( PRO_MINHEIGHT, minreqy )
            # check boundaries
            rn=self.createRnAddOthers()
            if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_PROCESS, aFullID, x, y, self )
            else:
                # change cursor
                self.theShape.setCursor(CU_CROSS)

        elif buttonPressed == PE_VARIABLE:
            # create command
            aName = self.getModelEditor().getUniqueEntityName ( ME_VARIABLE_TYPE, aSysPath )
            aFullID = ':'.join( [ME_VARIABLE_TYPE, aSysPath, aName] )
            objectID = self.theLayout.getUniqueObjectID( OB_TYPE_VARIABLE)
            
            minreqx, minreqy = self.getMinDims( ME_VARIABLE_TYPE, aFullID.split(":")[2] )
            x2=x+max( VAR_MINWIDTH, minreqx )
            y2=y+max( VAR_MINHEIGHT, minreqy )
            # check boundaries
            rn=self.createRnAddOthers()
            if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_VARIABLE, aFullID, x, y, self )
            else:
                # change cursor
                self.theShape.setCursor(CU_CROSS)


        elif buttonPressed == PE_TEXT:
            pass
            '''
            #aName = self.getModelEditor().getUniqueEntityName (ME_SYSTEM_TYPE, aSysPath )
            objectID = self.theLayout.getUniqueObjectID( OB_TYPE_TEXT )

            x2=x+TEXT_MINWIDTH
            y2=y+TEXT_MINHEIGHT
            # check boundaries
            rn=self.createRnAddSystem()
            if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_TEXT,None,x, y, self)
            else:
                # change cursor
                self.theShape.setCursor(CU_CROSS)
            '''
        elif buttonPressed == PE_SELECTOR:
            self.doSelect()
        elif buttonPressed == PE_CUSTOM:
            pass

        if aCommand != None:
            self.theLayout.passCommand( [aCommand] )

    def getObjectList( self ):
        # return IDs
        return self.theObjectMap.keys()


    def getObject( self, anObjectID ):
        return self.theObjectMap[ anObjectID ]
        
    def isWithinSystem( self, objectID ):
        #returns true if is within system
        pass
        
    def getAbsoluteInsidePosition( self ):
        ( x, y ) = self.getAbsolutePosition()
        return ( x+ self.theSD.insideX, y+self.theSD.insideY )
    
    #def isResizeOk(self,x,y)):

    def getCursorType( self, aFunction, x, y, buttonPressed ):

        maxposx=0;maxposy=0;maxnegx=0;maxnegy=0;maxpos=0;maxneg=0
        oneDirList=[DIRECTION_RIGHT,DIRECTION_UP,DIRECTION_DOWN,DIRECTION_LEFT]
        try:
            aCursorType = EditorObject.getCursorType( self, aFunction, x, y, buttonPressed )
            if aFunction == SD_SYSTEM_CANVAS and self.theLayout.getPaletteButton() != PE_SELECTOR:
                aCursorType = CU_ADD
            elif aFunction == SD_OUTLINE:
                olw=self.getProperty( OB_OUTLINE_WIDTH )
                direction = self.getDirection( x, y )
                if self.parentSystem.__class__.__name__ == 'Layout':
                    return self.cursorMap[direction]
                if direction in oneDirList:
                    maxpos=self.getMaxShiftPos(direction)
                    maxneg=self.getMaxShiftNeg(direction)
                    if maxpos>0 or maxneg>0 :
                        aCursorType = self.cursorMap[direction]
                    else:
                        aCursorType = CU_CROSS
                else:
                    maxposx,maxposy=self.getMaxShiftPos(direction)
                    maxnegx,maxnegy=self.getMaxShiftNeg(direction)
                    if maxposx>0 or maxposy>0 or maxnegx>0 or maxnegy>0  :
                        aCursorType = self.cursorMap[direction]
                    else:
                        aCursorType = CU_CROSS

                
        except:
            pass
        return aCursorType


    def getDirection( self, absx, absy ):
        olw = self.getProperty( OB_OUTLINE_WIDTH )
        width = self.getProperty( OB_DIMENSION_X )
        height = self.getProperty( OB_DIMENSION_Y )
        (offsetx, offsety ) = self.getAbsolutePosition()
        x = absx- offsetx
        y = absy - offsety

        direction = 0
        #leftwise direction:
        if x <= olw:
            direction |= DIRECTION_LEFT
        
        # rightwise direction
        elif x>= width -olw:
            direction |= DIRECTION_RIGHT

        # upwards direction
        if y <= olw:
            direction |= DIRECTION_UP
            

        # downwards direction
        elif y>= height - olw:
            direction |= DIRECTION_DOWN
            
            
        return direction

#-------------------------------------------------------------------
    def outlineDragged( self, deltax, deltay, absx, absy):
        theParent=self.parentSystem
        direction = self.getDirection( absx, absy )
        
        if not self.theShape.getIsButtonPressed() : # button released
            self.thedleftorg=self.thePropertyMap[OB_POS_X]-self.thex1org
            self.thedrightorg=self.thePropertyMap[OB_POS_X]+self.thePropertyMap[OB_DIMENSION_X]-self.thex2org
            self.theduporg=self.thePropertyMap[OB_POS_Y]-self.they1org
            self.theddownorg=self.thePropertyMap[OB_POS_Y]+self.thePropertyMap[OB_DIMENSION_Y]-self.they2org
            self.resize( self.theduporg, -self.theddownorg, self.thedleftorg, -self.thedrightorg)
            aCommand = ResizeObject( self.theLayout, self.theID, -self.theduporg, self.theddownorg, -self.thedleftorg, self.thedrightorg )

            # Layout
            newScrollRegion=self.theLayout.getProperty(LO_SCROLL_REGION)
            self.theLayout.setProperty(LO_SCROLL_REGION,self.theLayout.orgScrollRegion)
            self.theLayout.setProperty(OB_DIMENSION_X,self.theLayout.orgScrollRegion[2]-self.theLayout.orgScrollRegion[0])
            self.theLayout.setProperty(OB_DIMENSION_Y,self.theLayout.orgScrollRegion[3]-self.theLayout.orgScrollRegion[1])
            self.theLayout.getCanvas().setSize(self.theLayout.orgScrollRegion)
            aCommandLayout = ChangeLayoutProperty(self.theLayout, LO_SCROLL_REGION,newScrollRegion)
            self.theLayout.passCommand( [aCommand,aCommandLayout] )
            
        
        
        #FIXMEparentSystem boundaries should be watched!!!
        if self.theShape.getFirstDrag() and not self.theShape.getDragBefore() :
            self.thex1org=self.thePropertyMap[OB_POS_X]
            self.they1org=self.thePropertyMap[OB_POS_Y]
            self.thex2org=self.thex1org+self.thePropertyMap[OB_DIMENSION_X]
            self.they2org=self.they1org+self.thePropertyMap[OB_DIMENSION_Y]
            if theParent.__class__.__name__ != 'Layout':
                if direction==DIRECTION_UP or direction==DIRECTION_DOWN:
                    self.theMaxShiftPosy=self.getMaxShiftPos(direction)
                elif direction==DIRECTION_RIGHT or direction==DIRECTION_LEFT:
                    self.theMaxShiftPosx=self.getMaxShiftPos(direction)
                else:
                    self.theMaxShiftPosx,self.theMaxShiftPosy=self.getMaxShiftPos(direction)
            if direction==DIRECTION_UP or direction==DIRECTION_DOWN:
                self.theMaxShiftNegy=-(self.getMaxShiftNeg(direction))
            elif direction==DIRECTION_RIGHT or direction==DIRECTION_LEFT:
                self.theMaxShiftNegx=-(self.getMaxShiftNeg(direction))
            else:
                self.theMaxShiftNegx,self.theMaxShiftNegy=(self.getMaxShiftNeg(direction))
                self.theMaxShiftNegx=-self.theMaxShiftNegx
                self.theMaxShiftNegy=-self.theMaxShiftNegy
                
            lblWidth = self.theSD.getRequiredWidth()
            dimx = self.thePropertyMap[ OB_DIMENSION_X ]
            limitX = -(dimx-lblWidth)
            if limitX<0 and self.theMaxShiftNegx<limitX:
                self.theMaxShiftNegx=limitX

            self.theorgdir=direction
            self.theLayout.orgScrollRegion = self.theLayout.getProperty(LO_SCROLL_REGION)
            self.accshiftx=0;self.accshifty=0

            if direction not in [DIRECTION_RIGHT,DIRECTION_UP,DIRECTION_DOWN,DIRECTION_LEFT]:
                twoRectMat=self.getGraphUtils().buildTwoRect(self.thex1org,self.they1org,self.thex2org,self.they2org,self.theMaxShiftPosx,self.theMaxShiftPosy,self.theMaxShiftNegx,self.theMaxShiftNegy,self.theorgdir)
                self.prect1=n.array([twoRectMat[0][0],twoRectMat[0][1],twoRectMat[0][2],twoRectMat[0][3]])
                self.prect2=n.array([twoRectMat[1][0],twoRectMat[1][1],twoRectMat[1][2],twoRectMat[1][3]])
                self.rectdotx,self.rectdoty=self.getGraphUtils().getRectDotxy(self.thex1org,self.they1org,self.thex2org,self.they2org,self.theorgdir)
            
            
            self.theShape.setFirstDrag(False)
            self.theShape.setDragBefore(True)
            

        UDLR=self.getUDLRmatrix(deltax,deltay,self.theorgdir)
        deltaup=UDLR[self.UDLRMap['U']]
        deltadown=UDLR[self.UDLRMap['D']]
        deltaright=UDLR[self.UDLRMap['R']]
        deltaleft=UDLR[self.UDLRMap['L']]
        
        if direction!=self.theorgdir:
            direction=self.theorgdir
        
        
        if self.theShape.getIsButtonPressed() and self.theorgdir==direction:
            if direction in [DIRECTION_RIGHT,DIRECTION_UP,DIRECTION_DOWN,DIRECTION_LEFT]:
                if direction==DIRECTION_UP or direction==DIRECTION_DOWN:
                    if direction==DIRECTION_UP:
                        deltay=-deltay
                    newAccShift=self.accshifty+deltay
                    if newAccShift>0 and newAccShift<self.theMaxShiftPosy:
                        self.accshifty+=deltay
                        self.resize( deltaup, deltadown, deltaleft, deltaright)
                        if theParent.__class__.__name__ == 'Layout':
                            self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                        else:
                            self.adjustCanvas(deltax,deltay)
                    elif newAccShift<0 and newAccShift>self.theMaxShiftNegy:
                        self.accshifty+=deltay
                        self.resize( deltaup, deltadown, deltaleft, deltaright)
                        if theParent.__class__.__name__ == 'Layout':
                            self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                        else:
                            self.adjustCanvas(deltax,deltay)

                if direction==DIRECTION_LEFT or direction==DIRECTION_RIGHT:
                    if direction==DIRECTION_LEFT:
                        deltax=-deltax
                    newAccShift=self.accshiftx+deltax   
                    if newAccShift>0 and newAccShift<self.theMaxShiftPosx:
                        self.accshiftx+=deltax
                        self.resize( deltaup, deltadown, deltaleft, deltaright)
                        if theParent.__class__.__name__ == 'Layout':
                            self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                        else:
                            self.adjustCanvas(deltax,deltay)
                    elif newAccShift<0 and newAccShift>self.theMaxShiftNegx:
                        self.accshiftx+=deltax
                        self.resize( deltaup, deltadown, deltaleft, deltaright)
                        if theParent.__class__.__name__ == 'Layout':
                            self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                        else:
                            self.adjustCanvas(deltax,deltay)
            else:
                newdotx=self.rectdotx+deltax
                newdoty=self.rectdoty+deltay
                cond1=self.isWithinParent(newdotx,newdoty,newdotx,newdoty,self.prect1)
                cond2=self.isWithinParent(newdotx,newdoty,newdotx,newdoty,self.prect2)
                if cond1 and cond2:
                    self.rectdotx=newdotx
                    self.rectdoty=newdoty
                    self.resize( deltaup, deltadown, deltaleft, deltaright)
                    if theParent.__class__.__name__ == 'Layout':
                        self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                    else:
                        self.adjustCanvas(deltax,deltay)
                else:
                    return
            
    def getAvailableSystemShape(self):
            return self.theSystemShapeList

    def getLargestChildPosX(self):
        childProp=[]
        childs=self.getObjectList()
        for ch in childs:
            achild=self.getObject(ch)
            childProp.append(achild.getProperty(OB_POS_X)+achild.getProperty(OB_DIMENSION_X))
        childProp.sort()
        return childProp[len(childProp)-1]
        
    
    def createRnOut(self):
        no=len(self.parentSystem.getObjectList())
        rn=None
        if no>1:
            for sib in self.parentSystem.getObjectList():
                asib=self.parentSystem.getObject(sib)
                if asib.getProperty(OB_TYPE)!=OB_TYPE_CONNECTION and asib.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):
                    asibx1=asib.getProperty(OB_POS_X)
                    asiby1=asib.getProperty(OB_POS_Y)
                    asibx2=asibx1+asib.getProperty(OB_DIMENSION_X)
                    asiby2=asiby1+asib.getProperty(OB_DIMENSION_Y)
                    rsib=n.array([asibx1,asiby1,asibx2,asiby2])
                    rsib=n.reshape(rsib,(4,1))
                    if rn==None:
                        rn=rsib
                    else:
                        rn=n.concatenate((rn,rsib),1)
        return rn
    
    def createRnIn(self):
        no=len(self.getObjectList())
        rn=None
        olw=self.getProperty(OB_OUTLINE_WIDTH)
        if no>0:
            for ch in self.getObjectList():
                ach=self.getObject(ch)
                achx1=ach.getProperty(OB_POS_X)
                achy1=ach.getProperty(OB_POS_Y)
                achx2=achx1+ach.getProperty(OB_DIMENSION_X)
                achy2=achy1+ach.getProperty(OB_DIMENSION_Y)+8*olw
                rch=n.array([achx1,achy1,achx2,achy2])
                rch=n.reshape(rch,(4,1))
                if rn==None:
                    rn=rch
                else:
                    rn=n.concatenate((rn,rch),1)
        return rn

    def createRnAddSystem(self):
        no=len(self.getObjectList())
        rn=None
        if no>0:
            for sib in self.getObjectList():
                asib=self.getObject(sib)
                asibx1=asib.getProperty(OB_POS_X)
                asiby1=asib.getProperty(OB_POS_Y)
                asibx2=asibx1+asib.getProperty(OB_DIMENSION_X)
                asiby2=asiby1+asib.getProperty(OB_DIMENSION_Y)
                rsib=n.array([asibx1,asiby1,asibx2,asiby2])
                rsib=n.reshape(rsib,(4,1))
                if rn==None:
                    rn=rsib
                else:
                    rn=n.concatenate((rn,rsib),1)
        return rn

    def createRnAddOthers(self):
        no=len(self.getObjectList())
        rn=None
        if no>0:
            for sib in self.getObjectList():
                asib=self.getObject(sib)
                if asib.getProperty(OB_TYPE)==OB_TYPE_SYSTEM:
                    asibx1=asib.getProperty(OB_POS_X)
                    asiby1=asib.getProperty(OB_POS_Y)
                    asibx2=asibx1+asib.getProperty(OB_DIMENSION_X)
                    asiby2=asiby1+asib.getProperty(OB_DIMENSION_Y)
                    rsib=n.array([asibx1,asiby1,asibx2,asiby2])
                    rsib=n.reshape(rsib,(4,1))
                    if rn==None:
                        rn=rsib
                    else:
                        rn=n.concatenate((rn,rsib),1)
        return rn


    #######################################
    #  a system is being resized inwardly #
    #######################################
    def getMaxShiftNeg(self,direction):
        dir=direction
        x1=self.getProperty(OB_POS_X)
        y1=self.getProperty(OB_POS_Y)
        #x1=0;y1=0
        x2=x1+self.getProperty(OB_DIMENSION_X)
        y2=y1+self.getProperty(OB_DIMENSION_Y)
        r1=n.array([x1,y1,x2,y2])
        rn=self.createRnIn()
        textWidth=self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].getRequiredWidth()
        textHeight=self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].getRequiredHeight()
        if rn==None:
            olw=self.thePropertyMap[ OB_OUTLINE_WIDTH ] 
            u1=x1+(self.getProperty(OB_DIMENSION_X)-textWidth)
            v1=y1+(y2-y1-textHeight)
            u2=x2-(self.getProperty(OB_DIMENSION_X)-textWidth)
            v2=y2-(y2-y1-textHeight)
            matrix=n.array([u1,v1,u2,v2])
        else:
            matrix = self.getGraphUtils().calcMaxShiftNeg(r1,rn,dir)
        mshift=matrix-r1
        if len(self.maxShiftMap[dir])>1:
            posx,posy=self.maxShiftMap[dir][0],self.maxShiftMap[dir][1]
            return abs( mshift[posx] ),abs( mshift[posy] )
        else:
            pos=self.maxShiftMap[dir][0]
            return abs( mshift[pos] )
        
    #########################################
    #   Up Down Left Right Matrix       #
    #########################################
    def getUDLRmatrix(self,dx,dy,dir):
        m=n.array([dx,dy])
        m=n.reshape(m,(2,1))
        m=n.dot(self.dragMap[dir],m)
        m=n.reshape(m,(4,))
        if dir==DIRECTION_LEFT:
            m[self.UDLRMap['L']]=-m[self.UDLRMap['L']]
        if dir==DIRECTION_UP:
            m[self.UDLRMap['U']]=-m[self.UDLRMap['U']]
        return m

    def getDelta(self,UDLRmatrix):
        m=UDLRmatrix
        pos=n.nonzero(m)
        delta=0
        if n.size(pos)==1:
            delta=m[pos[0]]
            return delta
        if n.size(pos)==0:
            return delta
        mat=n.array([m[pos[0]],m[pos[1]]])
        if self.getGraphUtils().allGreater(mat):
            return max(m[pos[0]],m[pos[1]])
        elif self.getGraphUtils().allSmaller(mat):  
            return min(m[pos[0]],m[pos[1]])
        else:
            return max(abs(m[pos[0]]),abs(m[pos[1]]))
        
        
    def getAvailSpace(self,x,y,x2,y2,rn):
        px=0
        py=0
        px2=self.getProperty(OB_DIMENSION_X)
        py2=self.getProperty(OB_DIMENSION_Y)
        rpar=n.array([px,py,px2,py2])
        r1=n.array([x,y,x2,y2])
        dir=DIRECTION_BOTTOM_RIGHT
        matrix=self.getGraphUtils().calcMaxShiftPos(r1,rn,dir,rpar)
        mspace=matrix-r1
        mshift = n.array( [-1,-1,1,1] )
        mspace = mspace * mshift
        if len(self.maxShiftMap[dir])==1:
            listpos=self.maxShiftMap[dir]
            pos=listpos[0]
            return max(0,mspace[pos])
        else:
            listpos=self.maxShiftMap[dir]
            posa=listpos[0]
            posb=listpos[1]
            spacea=max(0,mspace[posa])
            spaceb=max(0,mspace[posb])
            return max(spacea,spaceb)

    ############################################### cheCk        
    def getSystemObject(self):
        return self
    ###############################################
    
    
