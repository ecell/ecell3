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

from ecell.ui.model_editor.EditorObject import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapeDescriptor import *
from ecell.ui.model_editor.LayoutCommand import *
from ecell.ui.model_editor.Utils import *

#from Process import *
from ecell.ui.model_editor.ShapePluginManager import *

OB_SHOW_LABEL=1

class ProcessObject( EditorObject ):
    def __init__( self, aLayout, objectID, aFullID,  x,y, canvas= None ):
        EditorObject.__init__( self, aLayout, objectID,x, y, canvas )
        self.thePropertyMap [ OB_FILL_COLOR ] = self.theLayout.graphUtils().getRRGByName("grey")

        self.thePropertyMap[ OB_HASFULLID ] = True
        self.thePropertyMap [ OB_FULLID ] = aFullID

        self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
        self.thePropertyMap[ OB_TYPE ] = OB_TYPE_PROCESS
        self.thePropertyMap[ PR_CONNECTIONLIST ] = []
        #default dimensions
        self.thePropertyMap [ OB_LABEL ]=aFullID.split(':')[2]
        self.theLabel = self.thePropertyMap [ OB_LABEL ]
        self.thePropertyMap [ OB_MINLABEL ]=PRO_MINLABEL
        aProcessSD=EditorObject.getShapeDescriptor(self, self.getProperty( OB_SHAPE_TYPE ) )
        #aProcessSD = ProcessSD(self, self.getGraphUtils(), self.theLabel )
        # first get text width and heigth

        reqWidth = aProcessSD.getRequiredWidth()
        reqHeight = aProcessSD.getRequiredHeight()
        
    
        self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
        if reqWidth<PRO_MINWIDTH:
            self.thePropertyMap [ OB_DIMENSION_X ]=PRO_MINWIDTH
        
        self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
        if reqHeight<PRO_MINHEIGHT:
            self.thePropertyMap [ OB_DIMENSION_Y ]=PRO_MINHEIGHT

        self.connectionDragged = False
        aProcessSD.reCalculate()
        self.setShapeDescriptor( aProcessSD )

    def registerConnection( self, aConnectionID ):
        self.thePropertyMap[ PR_CONNECTIONLIST ].append( aConnectionID )

    def unRegisterConnection( self, aConnectionID ):
        self.thePropertyMap[PR_CONNECTIONLIST].remove( aConnectionID )

    def getAvailableProcessShape(self):
        return self.theProcessShapeList

    def destroy( self ):
        connList = self.thePropertyMap[ PR_CONNECTIONLIST ][:]
        for aConnID in connList:
            self.theLayout.deleteObject( aConnID )
        EditorObject.destroy( self )

    def parentMoved( self, deltax, deltay ):
        EditorObject.parentMoved( self, deltax, deltay )

        for aConID in self.thePropertyMap[PR_CONNECTIONLIST]:
            self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

    def move( self, deltax, deltay ):
        EditorObject.move( self, deltax,deltay)
        for aConID in self.thePropertyMap[PR_CONNECTIONLIST]:
            self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

    def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
        #first do a resize then a move
        # FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
        # resize must be sum of deltas 
        self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
        self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown 
        self.theShape.resize(deltaleft + deltaright,deltaup + deltadown )

    def setShapeDescriptor( self, anSD ):
        EditorObject.setShapeDescriptor( self, anSD )
        for aConnectionID in self.thePropertyMap[ PR_CONNECTIONLIST]:
            aConnectionObject = self.theLayout.getObject( aConnectionID )
            aConnectionObject.reconnect()
        
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
            ( variableID, variableRing ) = self.theLayout.checkConnection( endx, endy, ME_VARIABLE_TYPE )
            newVarrefName = None
            if variableID == None:
                aCommand , width, height = self.theLayout.thePackingStrategy.createEntity( ME_VARIABLE_TYPE, endx, endy )
                # create totally new variable
                if aCommand == None:
                    self.theShape.setCursor( CU_CROSS )
                    variableID = None
                else:
                    ringSource, ringDest = self.theLayout.thePackingStrategy.autoConnect( self.getID(), (endx+width/2, endy+height/2) )
                    newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                    newVarrefName = self.__getNewVarrefID ()

                    connectCommand = CreateConnection( self.theLayout, newID, self.theID, aCommand.getID(), self.theRingCode, ringDest, PROCESS_TO_VARIABLE, newVarrefName )
                    self.theLayout.passCommand( [ aCommand] )
                    self.theLayout.passCommand( [ connectCommand ] )
                    variableID = aCommand.getID()
            # create real line
            else:
                # check for already existing varref
                varObject = self.theLayout.getObject(variableID)
                variableFullID = varObject.getProperty(OB_FULLID)
                relFullID = getRelativeReference(self.getProperty(OB_FULLID), variableFullID)
                #get the process varreflist
                aProFullPN = createFullPN (self.getProperty(OB_FULLID), MS_PROCESS_VARREFLIST )
                aVarrefList = copyValue( self.getModelEditor().getModel().getEntityProperty( aProFullPN) )
                #filter aVarrefList by variableFullID
                aSpecVarrefList=[]
                for aVarref in aVarrefList:
                    if aVarref[ME_VARREF_FULLID] in (variableFullID,relFullID):
                        aSpecVarrefList+=[aVarref]
            
                #get the pro connection obj  
                connectionList = self.getProperty(PR_CONNECTIONLIST)
                displayedVarrefList=[]
                aSpecDisplayedVarrefList=[]
                for conn in connectionList:
                    connObj = self.theLayout.getObject( conn )
                    varreffName = connObj.getProperty(CO_NAME)
                    varID = connObj.getProperty(CO_VARIABLE_ATTACHED)
                    #get var FUllID
                    if varID != None:
                        varFullID = self.theLayout.getObject( varID ).getProperty(OB_FULLID)
                        displayedVarrefList += [ [ varreffName, varFullID ] ]
                for aVarref in displayedVarrefList:
                    if aVarref[ME_VARREF_FULLID] == variableFullID:
                        aSpecDisplayedVarrefList += [ aVarref [ ME_VARREF_NAME ] ]
                
                
                #check if there is existing varref that hasn't been displayed
                if len(aSpecVarrefList)!=len(aSpecDisplayedVarrefList) :
                    for aVarref in aSpecVarrefList:
                        if aVarref[ME_VARREF_NAME] not in aSpecDisplayedVarrefList:
                            newVarrefName = aVarref[ME_VARREF_NAME]
            
            
                newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                if newVarrefName == None:
                    newVarrefName = self.__getNewVarrefID ()
                aCommand = CreateConnection( self.theLayout, newID, self.theID, variableID, self.theRingCode, variableRing, PROCESS_TO_VARIABLE, newVarrefName )
                self.theLayout.passCommand( [ aCommand ] )
            if variableID != None:
                self.theLayout.selectRequest( variableID )
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

    def estLabelWidth(self,newLabel):
        #height,width=self.getGraphUtils().getTextDimensions(newLabel)
        #return width+2+9
        return self.theSD.estLabelWidth(newLabel)

    def labelChanged( self,aPropertyValue ):
        #newLabel = aPropertyValue.split(':')[2]
        newLabel = aPropertyValue
        #totalWidth,limit=self.getLabelParam()
        #if totalWidth>limit:
        #   newLabel=self.truncateLabel(newLabel,totalWidth,limit)
        #   self.thePropertyMap[OB_LABEL]=newLabel
        self.theShape.labelChanged(self.getProperty(OB_LABEL)) 
        
        if self.thePropertyMap[ PR_CONNECTIONLIST ] !=[]:
                
            for conn in self.getProperty(PR_CONNECTIONLIST):
                conobj =  self.theLayout.getObject(conn)
                (x, y) = self.getRingPosition(conobj.thePropertyMap[ CO_PROCESS_RING ] )
                rsize = self.getRingSize()
                # this shoould be done by connection object enpoint1chgd
                conobj.thePropertyMap[ CO_ENDPOINT1 ] = [ x +rsize/2, y+rsize/2 ]
                conobj.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].reCalculate()
                conobj.theShape.repaint()

    def __getNewVarrefID( self ):
        aFullID = self.getProperty( OB_FULLID )
        aFullPN = createFullPN ( aFullID, ME_PROCESS_VARREFLIST )
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


