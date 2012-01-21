#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
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

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.EditorObject import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapeDescriptor import *
from ecell.ui.model_editor.LayoutCommand import *
from ecell.ui.model_editor.ShapePluginManager import *

OB_SHOW_LABEL=1

class VariableObject( EditorObject ):
    
    def __init__( self, aLayout,objectID, aFullID,  x,y, canvas= None ):
        EditorObject.__init__( self, aLayout, objectID, x, y, canvas)
        self.thePropertyMap[ OB_HASFULLID ] = True
        self.thePropertyMap [ OB_FULLID ] = aFullID
        self.theObjectMap = {}
        self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
        self.thePropertyMap[ OB_TYPE ] = OB_TYPE_VARIABLE
        #default dimensions
        self.thePropertyMap [ OB_LABEL ]=aFullID.split(':')[2]
        self.thePropertyMap [ OB_MINLABEL ]=VAR_MINLABEL
        aVariableSD=EditorObject.getShapeDescriptor(self, self.getProperty( OB_SHAPE_TYPE ) )
        # first get text width and heigth
        self.thePropertyMap[ VR_CONNECTIONLIST ]= []
        reqWidth = aVariableSD.getRequiredWidth()
        reqHeight = aVariableSD.getRequiredHeight()
        self.connectionDragged = False

        self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
        if reqWidth<VAR_MINWIDTH:
            self.thePropertyMap [ OB_DIMENSION_X ]=VAR_MINWIDTH


        self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
        if reqHeight<VAR_MINHEIGHT:
            self.thePropertyMap [ OB_DIMENSION_Y]=VAR_MINHEIGHT
        aVariableSD.reCalculate()
        self.setShapeDescriptor( aVariableSD )
        
        
    def reconnect( self ):
        pass

    def destroy( self ):
        connList = self.thePropertyMap[ VR_CONNECTIONLIST ][:]
        for aConnID in connList:
            self.theLayout.deleteObject( aConnID )
        EditorObject.destroy( self )


    def registerConnection( self, aConnectionID ):
        self.thePropertyMap[ VR_CONNECTIONLIST ].append( aConnectionID )

    def unRegisterConnection( self, aConnectionID ):
        
        self.thePropertyMap[ VR_CONNECTIONLIST].remove( aConnectionID )


    def setShapeDescriptor( self, anSD ):
        EditorObject.setShapeDescriptor( self, anSD )
        for aConnectionID in self.thePropertyMap[ VR_CONNECTIONLIST]:
            aConnectionObject = self.theLayout.getObject( aConnectionID )
            aConnectionObject.reconnect()


    def estLabelWidth(self,newLabel):
        #height,width=self.getGraphUtils().getTextDimensions(newLabel)
        #return width+46
        return self.theSD.estLabelWidth(newLabel)

    def labelChanged( self,aPropertyValue ):
        #newLabel = aPropertyValue.split(':')[2]
        newLabel = aPropertyValue
        #totalWidth,limit=self.getLabelParam()
        #if totalWidth>limit:
        #   newLabel=self.truncateLabel(newLabel,totalWidth,limit)
        #   self.thePropertyMap[OB_LABEL]=newLabel
        self.theShape.labelChanged(self.getProperty(OB_LABEL))  
        
        

    def getAvailableVariableShape(self):
        return self.theVariableShapeList

    def parentMoved( self, deltax, deltay ):
        EditorObject.parentMoved( self, deltax, deltay )
        for aConID in self.thePropertyMap[VR_CONNECTIONLIST]:
            self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )


    def move( self, deltax, deltay ):
        EditorObject.move( self, deltax,deltay)
        for aConID in self.thePropertyMap[VR_CONNECTIONLIST]:
            self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

    def registerObject( self, anObject ):
        self.theObjectMap[anObject.getID()] = anObject


    def unregisterObject ( self, anObjectID ):
        self.theObjectMap.__delitem__( anObjectID )


    def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
        #first do a resize then a move
        # FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
        # resize must be sum of deltas
        self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
        self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown

        self.theShape.resize(deltaleft + deltaright,deltaup + deltadown)
        if self.thePropertyMap[ VR_CONNECTIONLIST ] !=[]:
                
            for conn in self.getProperty(VR_CONNECTIONLIST):
                conobj =  self.theLayout.getObject(conn)
                (x, y) = self.getRingPosition(conobj.thePropertyMap[ CO_VARIABLE_RING ] )
                ringsize =  self.theSD.getRingSize()/2
                conobj.thePropertyMap[ CO_ENDPOINT2 ] = [ x + ringsize, y+ringsize]
                conobj.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].reCalculate()
                conobj.theShape.repaint()



    def buttonReleased( self ):
        EditorObject.buttonReleased( self )
        if self.connectionDragged:
            self.connectionDragged = False
            # get coordinates
            ( endx, endy) = self.theGhostLine.getEndPoint( )
            self.theGhostLine.delete()
            self.theGhostLine = None
            # delete ghostline
            # CHECK IF VARIABLE IS CONNECTED
            ( processID, processRing ) = self.theLayout.checkConnection( endx, endy, ME_PROCESS_TYPE )
            newVarrefName = None
            if processID == None:
                cmdList = []
                # create a Process and a Variable
                ( varID, varRing ) = self.theLayout.checkConnection( endx, endy, ME_VARIABLE_TYPE )
                if varID == None:
                    varx, vary = endx, endy
                    varCommand , varwidth, varheight = self.theLayout.thePackingStrategy.createEntity( ME_VARIABLE_TYPE, varx, vary )
                    if varCommand == None:
                        self.theShape.setCursor( CU_CROSS )
                        return
                    varID = varCommand.getID()
                    cmdList.append( varCommand )
                else:
                    varObject = self.theLayout.getObject( varID )
                    varx, vary = varObject.getAbsolutePosition()
                    varwidth, varheight = varObject.getProperty( OB_DIMENSION_X), varObject.getProperty( OB_DIMENSION_Y )
                i = 0
                startx, starty = self.getAbsolutePosition()
                startx += self.getProperty(OB_DIMENSION_X)/2
                starty += self.getProperty( OB_DIMENSION_Y )/2
                varx += varwidth/2
                vary += varheight/2
                diffx, diffy = varx - startx, vary - starty
                procCommand = None
                while i != 5 and procCommand == None :
                    procx, procy = startx + (i + 5) * diffx/10, starty + (i + 5) *diffy/10
                    procCommand, procwidth, procheight = self.theLayout.thePackingStrategy.createEntity( ME_PROCESS_TYPE, procx, procy )
                    i *= -1
                    if i >= 0:
                        i +=1
                if procCommand == None:
                    self.theShape.setCursor( CU_CROSS )
                    return
                procID =  procCommand.getID()
                cmdList.append( procCommand )
                self.theLayout.passCommand( cmdList )
                # create lines between self, newprocess, newprocess, newvariable
                procObject = self.theLayout.getObject( procID )
                ringSource, ringDest = self.theLayout.thePackingStrategy.autoConnect( self.getID(), procID )
                firstID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                firstVarrefName = self.__getNewVarrefID (procObject.getProperty(OB_FULLID ) )
                firstConnectCommand = CreateConnection( self.theLayout, firstID,  procID, self.theID,  ringDest, self.theRingCode, VARIABLE_TO_PROCESS, firstVarrefName )
                
                ringSource, ringDest = self.theLayout.thePackingStrategy.autoConnect( procID, varID )
                secondID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                secondVarrefName = firstVarrefName + "1"
                secondConnectCommand = CreateConnection( self.theLayout, secondID, procID, varID, ringSource, ringDest, PROCESS_TO_VARIABLE, secondVarrefName )
                self.theLayout.passCommand( [firstConnectCommand] )
                self.theLayout.passCommand( [secondConnectCommand] )
                self.theLayout.selectRequest( varID )
                
                
                
            else:    
                # create real line
                variableFullID = self.getProperty( OB_FULLID )
                procObject = self.theLayout.getObject(processID)
                processFullID = procObject.getProperty(OB_FULLID)
                relFullID = getRelativeReference( processFullID, variableFullID )
                #get the process varreflist
                aProFullPN = createFullPN (processFullID, MS_PROCESS_VARREFLIST )
                aVarrefList =  self.getModelEditor().getModel().getEntityProperty( aProFullPN) 
                #filter aVarrefList by variableFullID
                aSpecVarrefList=[]
                for aVarref in aVarrefList:
                    if aVarref[ME_VARREF_FULLID] in (variableFullID, relFullID):
                        aSpecVarrefList+=[aVarref]
            
                #get the pro connection obj  
                connectionList = procObject.getProperty(PR_CONNECTIONLIST)

                aSpecDisplayedVarrefList=[]
                for conn in connectionList:
                    connObj = self.theLayout.getObject( conn )
                    varreffName = connObj.getProperty(CO_NAME)
                    varID = connObj.getProperty(CO_VARIABLE_ATTACHED)
                    #get var FUllID
                    if varID!=None:
                        varFullID = self.theLayout.getObject( varID ).getProperty(OB_FULLID)
                        if varFullID == variableFullID:
                            aSpecDisplayedVarrefList+=[varreffName]
                
                
                #check if there is existing varref that hasn't been displayed
                if len(aSpecVarrefList)!=len(aSpecDisplayedVarrefList) :
                    for aVarref in aSpecVarrefList:
                        if aVarref[ME_VARREF_NAME] not in aSpecDisplayedVarrefList:
                            newVarrefName = aVarref[ME_VARREF_NAME]
                else:
                    newVarrefName = self.__getNewVarrefID ( processFullID )

                newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                aCommand = CreateConnection( self.theLayout, newID, processID, self.theID, 
                        processRing,self.theRingCode,  VARIABLE_TO_PROCESS, newVarrefName )
                self.theLayout.passCommand( [ aCommand ] )
                # newCon = self.theLayout.getObject( newID )
                # newCon.checkConnections()
            


    def ringDragged( self, aShapeName, deltax, deltay ):
        if self.connectionDragged:
            self.theGhostLine.moveEndPoint( deltax, deltay )
        else:
            self.theRingCode = aShapeName
            
            ( x, y ) = self.getRingPosition( aShapeName )
            x += self.theSD.getRingSize()/2
            y += self.theSD.getRingSize()/2
            self.theGhostLine = GhostLine( self, aShapeName, x+deltax, y+deltay )
            self.connectionDragged = True

      
    def getRingSize( self ):
        return self.theSD.getRingSize()


    def getRingPosition( self, ringCode ):
        #return absolute position of ring
        (xRing,yRing)=self.theSD.getShapeAbsolutePosition(ringCode)
        ( x, y ) = self.getAbsolutePosition()
        return (x+xRing, y+yRing )



    def __getNewVarrefID( self, aProcessFullID ):
        aFullPN = createFullPN ( aProcessFullID, ME_PROCESS_VARREFLIST )
        aVarrefList = self.theLayout.theLayoutManager.theModelEditor.getModel().getEntityProperty( aFullPN )
        aVarrefNameList = []
        for aVarref in aVarrefList:
            aVarrefNameList.append( aVarref[ME_VARREF_NAME] )
        i= 0
        while 'X' +str(i) in aVarrefNameList:
            i += 1
        return 'X' + str( i )

    def getCursorType( self, aFunction, x, y, buttonPressed ):
        cursorType = EditorObject.getCursorType( self,  aFunction, x, y, buttonPressed  )
        if aFunction == SD_RING and buttonPressed:
            return CU_CONN_INIT
        return cursorType 
