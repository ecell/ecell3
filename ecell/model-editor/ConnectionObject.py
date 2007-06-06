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
from EditorObject import *
from Constants import *
from LineDescriptor import *
from LayoutCommand import *
from Utils import *
from ComplexLine import *

class ConnectionObject( EditorObject ):
    
    def __init__( self, aLayout,objectID, aVariableID, aProcessID, aVariableRing, aProcessRing, aVarrefName, parentSystem ):
        EditorObject.__init__( self, aLayout, objectID, 0, 0, parentSystem)
        self.thePropertyMap[ CO_PROCESS_RING ] = aProcessRing
        self.thePropertyMap[ CO_VARIABLE_RING ] = aVariableRing

        self.thePropertyMap[ OB_HASFULLID ] = False
        self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_STRAIGHT_LINE
        self.thePropertyMap [ CO_LINEWIDTH ] = 3
        self.thePropertyMap[ OB_TYPE ] = OB_TYPE_CONNECTION
        self.thePropertyMap [ OB_FILL_COLOR ] = self.theLayout.graphUtils().getRRGByName("black")
        self.thePropertyMap[ CO_CONTROL_POINTS ] = None

        #default dimensions
        # get label from processID
        processObj = self.theLayout.getObject( aProcessID )
        aFullID = processObj.getProperty( OB_FULLID )
        aVarrefList = self.theLayout.theLayoutManager.theModelEditor.getModel().getEntityProperty( createFullPN( aFullID, ME_PROCESS_VARREFLIST ) )
        aCoef = 0
        for aVarref in aVarrefList:
            if aVarref[ MS_VARREF_NAME ] == aVarrefName:
                aCoef = int(aVarref[ MS_VARREF_COEF ])


        self.thePropertyMap[ CO_NAME ] = aVarrefName
        self.thePropertyMap[ CO_COEF ] = aCoef
        self.thePropertyMap[ CO_USEROVERRIDEARROW ] = False
        if type(aProcessID) == type( [] ): #CANNOT BE!!!
            self.thePropertyMap[ CO_PROCESS_ATTACHED ] = None
            self.thePropertyMap[ CO_ENDPOINT1 ] = aProcessID
            self.thePropertyMap[ CO_ATTACHMENT1TYPE ] = OB_NOTHING
            self.thePropertyMap[ CO_DIRECTION1 ] = self.__getRingDirection( RING_TOP )
        else:
            self.thePropertyMap[ CO_PROCESS_ATTACHED ] = aProcessID
            aProcessObj = self.theLayout.getObject ( aProcessID )
            aProcessObj.registerConnection( objectID )
            self.thePropertyMap[ CO_ENDPOINT1 ] = self.getRingPosition( aProcessID, aProcessRing )
            self.thePropertyMap[ CO_ATTACHMENT1TYPE ] = OB_TYPE_PROCESS
            aProcessFullID = aProcessObj.getProperty( OB_FULLID )
            aModelEditor = self.theLayout.theLayoutManager.theModelEditor
            aVarrefList = aModelEditor.getModel().getEntityProperty( aProcessFullID + ':' + ME_PROCESS_VARREFLIST )
            for aVarref in aVarrefList:
                if aVarref[ MS_VARREF_NAME ] == aVarrefName:
                    self.thePropertyMap[ CO_COEF ] = int(aVarref[ MS_VARREF_COEF ])
                    break
            self.thePropertyMap[ CO_DIRECTION1 ] = self.__getRingDirection( aProcessRing )
            
        if type(aVariableID) == type( [] ):
            self.thePropertyMap[ CO_VARIABLE_ATTACHED ] = None
            self.thePropertyMap[ CO_ENDPOINT2 ] = aVariableID
            self.thePropertyMap[ CO_ATTACHMENT2TYPE ] = OB_NOTHING
            self.thePropertyMap[ CO_DIRECTION2 ] = self.__getRingDirection( RING_BOTTOM)
        else:
            self.thePropertyMap[ CO_VARIABLE_ATTACHED ] = aVariableID
            aVariableObj = self.theLayout.getObject( aVariableID )
            self.thePropertyMap[ CO_ENDPOINT2 ] = self.getRingPosition( aVariableID, aVariableRing )
            self.thePropertyMap[ CO_ATTACHMENT2TYPE ] = OB_NOTHING
            self.thePropertyMap[ CO_DIRECTION2 ] = self.__getRingDirection( aVariableRing)
            self.thePropertyMap[ CO_ATTACHMENT2TYPE ] = OB_TYPE_VARIABLE
            aVariableObj.registerConnection( objectID )
        self.__defineArrowDirection()

        aLineSD = StraightLineSD(self, self.getGraphUtils() )

        self.theSD = aLineSD
        self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aLineSD
        self.theConnectionArrowTypeList=['Straight','Cornered', 'Curved','MultiBezierCurve']
        self.theConnectionLineTypeList=['Normal', 'Bold', 'Dashed' ,'Dotted']
        self.hasBeenDragBefore = False
        #Attribute needed for redirectCon
        self.EndPoint2 =self.thePropertyMap[ CO_ENDPOINT2 ]
        self.processRing = self.thePropertyMap[ CO_PROCESS_RING ]       

    def __defineArrowDirection( self ):
        self.thePropertyMap[ CO_HASARROW1 ] = False
        self.thePropertyMap[ CO_HASARROW2 ] = False
        if self.thePropertyMap[ CO_COEF ] == 0:
            return
        elif self.thePropertyMap[ CO_COEF ] <0:
            self.thePropertyMap[ CO_HASARROW1 ] = True
        else:
            self.thePropertyMap[ CO_HASARROW2 ] = True

    def arrowheadDragged(self,shapeName, deltax, deltay, absx, absy):
        (offsetx, offsety ) = self.getAbsolutePosition()
        x = absx - offsetx
        y = absy - offsety
        if self.outOfRoot( x, y ):
            return 
        if self.theShape.getFirstDrag():
            self.redirectConnection(shapeName,x,y)
            self.theShape.setFirstDrag(False)
            self.hasBeenDragBefore = True

    def redirectConnection(self,shapeName,x,y):
        if shapeName == ARROWHEAD1:
            self.thePropertyMap[ CO_ENDPOINT1 ] = [x, y]
        elif shapeName == ARROWHEAD2:
            self.thePropertyMap[ CO_ENDPOINT2 ] = [x, y]
        self.theShape.repaint()

    def mouseReleased(self,shapeName, x, y):
        if self.theShape.getFirstDrag():
            self.arrowheadDragged(0, 0, x, y)
        
        if self.hasBeenDragBefore :
            if shapeName == ARROWHEAD2:
                ( varID, varRing ) = self.theLayout.checkConnection(x, y, ME_VARIABLE_TYPE )
                
                if varID == None:
                    varID = ( x, y)
                    varRing = None
                self.thePropertyMap[ CO_ENDPOINT2 ]  = self.EndPoint2
                aCommand = RedirectConnection( self.theLayout, self.theID, None, varID,None,varRing,None)
                self.theLayout.passCommand( [aCommand] )
                self.hasBeenDragBefore = False

            elif shapeName == ARROWHEAD1:
                ( proID, processRing ) = self.theLayout.checkConnection(x, y, ME_PROCESS_TYPE )
                if proID ==None or proID != self.thePropertyMap[ CO_PROCESS_ATTACHED ]:
                    proID = self.thePropertyMap[ CO_PROCESS_ATTACHED ]
                    processRing = self.thePropertyMap[ CO_PROCESS_RING ]
                    aProcessObj = self.theLayout.getObject( proID )
                    (x, y) = aProcessObj.getRingPosition( processRing )
                    rsize = aProcessObj.getRingSize()
                    self.thePropertyMap[ CO_ENDPOINT1 ] = [ x +rsize/2, y+rsize/2 ]
                    self.theShape.repaint()
                    return 
                else:
                    self.processRing =  processRing
                    aCommand = RedirectConnection( self.theLayout, self.theID, proID,None,processRing,None,None)
                    self.theLayout.passCommand( [aCommand] )
                    self.hasBeenDragBefore = False
            '''
            elif shapeName == SHAPE_TYPE_MULTIBCURVE_LINE:
                self.getArrowType(SHAPE_TYPE_MULTIBPATH_LINE)
                self.thePropertyMap[CO_CONTROL_POINTS] = theSD.theDescriptorList["MultiBezierCurve"][SD_SPECIFIC]
            '''    
                
                    


    def redirectConnbyComm(self, aProID,aNewVarID,aProcessRing,aVariableRing,varrefName):
        # arguments are None. means they dont change
        # if newVarId is like [x,y] then i should be detached
        
        if aNewVarID != None:
            # means it changed
            oldVarID = self.thePropertyMap[ CO_VARIABLE_ATTACHED ]
            if type(aNewVarID) in (type( [] ),type([] )) :
                self.thePropertyMap[ CO_ENDPOINT2 ] = aNewVarID
                self.thePropertyMap[ CO_VARIABLE_ATTACHED ] = None
                self.thePropertyMap[ CO_ATTACHMENT2TYPE ] = OB_NOTHING
                self.thePropertyMap[ CO_DIRECTION2 ] = self.__getRingDirection( RING_BOTTOM)
            else:
                self.thePropertyMap[ CO_VARIABLE_ATTACHED ] = aNewVarID
                aVariableObj = self.theLayout.getObject( aNewVarID)
                (x, y) = aVariableObj.getRingPosition( aVariableRing )

                rsize =  aVariableObj.theSD.getRingSize()/2
                self.thePropertyMap[ CO_ENDPOINT2 ] = [x + rsize, y+rsize]
                self.thePropertyMap[ CO_DIRECTION2 ] = self.__getRingDirection( aVariableRing)
                self.thePropertyMap[ CO_VARIABLE_RING ] = aVariableRing
                self.thePropertyMap[ CO_ATTACHMENT2TYPE ] = OB_TYPE_VARIABLE
                aVariableObj.registerConnection(self.theID)
                
            if oldVarID != None:
                self.theLayout.getObject( oldVarID ).unRegisterConnection( self.theID )

            self.theShape.repaint() 
            self.EndPoint2 = self.thePropertyMap[ CO_ENDPOINT2 ]        
        if aProID != None:
            
            if aProcessRing !=None:
                aProcessObj = self.theLayout.getObject( aProID)
                (x, y) = aProcessObj.getRingPosition( aProcessRing )
                rsize = aProcessObj.getRingSize()
                self.thePropertyMap[ CO_ENDPOINT1 ] = [ x +rsize/2, y+rsize/2 ]
                self.thePropertyMap[ CO_DIRECTION1 ] = self.__getRingDirection( aProcessRing )
            self.theShape.repaint() 
            self.thePropertyMap[  CO_PROCESS_RING ] = self.processRing


    def parentMoved( self, parentID, deltax, deltay ):
        if parentID ==self.thePropertyMap[ CO_VARIABLE_ATTACHED ]:
            changedType = OB_TYPE_VARIABLE
        else:
            changedType = OB_TYPE_PROCESS
        if changedType == self.thePropertyMap[ CO_ATTACHMENT1TYPE ]:
            chgProp = CO_ENDPOINT1
        else:
            chgProp = CO_ENDPOINT2
        (x, y) = self.thePropertyMap[ chgProp ]
        x+=deltax
        y+=deltay
        self.thePropertyMap[ chgProp ] = [x, y]
        self.theShape.repaint()


    def __getRingDirection( self, ringCode ):
        if ringCode == RING_TOP:
            return DIRECTION_UP
        elif ringCode == RING_BOTTOM:
            return DIRECTION_DOWN
        elif ringCode == RING_LEFT:
            return DIRECTION_LEFT
        elif ringCode == RING_RIGHT:
            return DIRECTION_RIGHT


    def reconnect( self ):
        aProcessID = self.getProperty( CO_PROCESS_ATTACHED )
        aProcessRing = self.getProperty( CO_PROCESS_RING )
        self.thePropertyMap[ CO_ENDPOINT1 ] = self.getRingPosition( aProcessID, aProcessRing )
        aVariableID = self.getProperty( CO_VARIABLE_ATTACHED )
        if aVariableID != None:
            aVariableRing = self.getProperty( CO_VARIABLE_RING )        
            self.thePropertyMap[ CO_ENDPOINT2 ] = self.getRingPosition( aVariableID, aVariableRing )
        self.theShape.repaint()


    def getRingPosition( self, anObjectID, aRing ):
            anObj = self.theLayout.getObject( anObjectID )
            (x, y) = anObj.getRingPosition( aRing )
            rsize = anObj.getRingSize()
            return [ x +rsize/2, y+rsize/2 ]


    def setProperty(self, aPropertyName, aPropertyValue):
        self.thePropertyMap[aPropertyName] = aPropertyValue
        if aPropertyName == OB_SHAPE_TYPE:
            if  self.theCanvas !=None:
                self.theShape.delete()
            self.getArrowType(aPropertyValue) #hereeee
            if  self.theCanvas !=None:
                self.theShape.show()
        elif aPropertyName == CO_NAME:
            if  self.theCanvas !=None:
                self.theShape.reName()
        elif aPropertyName == CO_COEF:
            self.__defineArrowDirection()
            if  self.theCanvas !=None:
                self.theShape.repaint()
        elif aPropertyName == CO_LINETYPE:
            pass
        elif aPropertyName == CO_LINEWIDTH:
            pass
        elif aPropertyName == CO_HASARROW1:
            pass
        elif aPropertyName == CO_HASARROW2:
            pass
        elif aPropertyName == OB_FILL_COLOR:
            if self.theCanvas != None:
                self.theShape.fillColorChanged( )


        # it should - gabor
        elif aPropertyName == CO_CONTROL_POINTS:
            if self.theCanvas != None:
                self.theShape.repaint()


    def getAvailableArrowType(self):
        return self.theConnectionArrowTypeList


    def getAvailableLineType(self):
        return self.theConnectionLineTypeList

    def getArrowType(self, aShapeType):
        if aShapeType == SHAPE_TYPE_STRAIGHT_LINE:
            aLineSD = StraightLineSD(self, self.getGraphUtils() )
        elif aShapeType == SHAPE_TYPE_CORNERED_LINE:
            aLineSD = corneredLineSD(self, self.getGraphUtils() )
        elif aShapeType == SHAPE_TYPE_CURVED_LINE:
            aLineSD = curvedLineSD(self, self.getGraphUtils() )
        elif aShapeType == SHAPE_TYPE_MULTIBCURVE_LINE:
            aLineSD = multiBcurveLineSD(self, self.getGraphUtils())
            
        self.theSD = aLineSD
        self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aLineSD

        ####### cheCk
        '''
        self.thePropertyMap[CO_CONTROL_POINTS] = theSD.theDescriptorList["MultiBezierCurve"][SD_SPECIFIC]
        '''

    def show( self ):
        self.theShape = ComplexLine( self, self.theCanvas )
        self.theShape.show()

    
    def checkConnections( self, end = 2 ):
        # get position of arrowhead 2
        queryProp = CO_ENDPOINT2
        attProp = CO_ATTACHMENT2TYPE

        if end ==1:
            queryProp = CO_ENDPOINT1
            attProp = CO_ATTACHMENT1TYPE
        (x,y) = self.thePropertyMap[ queryProp ]
        currentAttachment = self.thePropertyMap[ attProp ]
        if currentAttachment == OB_TYPE_PROCESS:
            checkFor = OB_TYPE_PROCESS
        else:
            checkFor = OB_TYPE_VARIABLE
        ( aShapeID, aRingID ) = self.theLayout.checkConnection( x, y, checkFor )


    def destroy(self):
        EditorObject.destroy( self )
        varID = self.thePropertyMap[ CO_VARIABLE_ATTACHED ]
        if varID != None:
            self.theLayout.getObject( varID ).unRegisterConnection( self.theID )
        procID = self.thePropertyMap[ CO_PROCESS_ATTACHED ]
        self.theLayout.getObject( procID ).unRegisterConnection( self.theID )
        
