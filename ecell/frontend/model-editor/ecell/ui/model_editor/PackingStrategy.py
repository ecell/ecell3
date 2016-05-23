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

import math 

from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.LayoutCommand import *

class PackingStrategy:
    angleMatrix = [ [RING_RIGHT, RING_LEFT], [RING_TOP, RING_BOTTOM], [RING_LEFT,RING_RIGHT], [RING_BOTTOM, RING_TOP] ]
    def __init__( self, aLayout ):
        self.theLayout = aLayout
        
    def autoMoveObject( self, systemFullID, objectID ):
        # return cmdlist: delete or move + resize command
        cmdList = []
        # get Object of systemFullID
        systemList = self.theLayout.getObjectList( OB_TYPE_SYSTEM )
        systemObject = None
        for aSystemObject in systemList:
            if aSystemObject.getProperty( OB_FULLID ) == systemFullID:
                systemObject = aSystemObject
                break
        if systemObject == None:
            # create delete command
            delCmd = DeleteObject ( self.theLayout, objectID )
            cmdList.append( delCmd )
        else:
            anObject = self.theLayout.getObject( objectID )
            # get dimensions of object
            objectWidth = anObject.getProperty( OB_DIMENSION_X )
            objectHeigth = anObject.getProperty( OB_DIMENSION_Y )
            
            # get inside dimensions of system
            systemWidth = systemObject.getProperty ( SY_INSIDE_DIMENSION_X )
            systemHeigth = systemObject.getProperty ( SY_INSIDE_DIMENSION_Y )

            # resize if necessary
            resizeNeeded = False
            oldObjectWidth = objectWidth
            oldObjectHeigth = objectHeigth
            if objectWidth >= systemWidth:
                resizeNeeded = True
                objectWidth = systemWidth /2
                if objectWidth < OB_MIN_WIDTH:
                    objectWidth = OB_MIN_WIDTH
        
            
            if objectHeigth >= systemHeigth:
                resizeNeeded = True
                objectHeigth = systemHeigth /2
                if objectHeigth < OB_MIN_HEIGTH:
                    objectHeigth = OB_MIN_HEIGTH

            if resizeNeeded:
                cmdList.append( ResizeObject( self.theLayout, objectID, 0, objectHeigth - oldObjectHeigth, 0, objectWidth - oldObjectWidth ) )
            # get rnd coordinates
            leewayX = systemWidth - objectWidth
            leewayY = systemHeigth - objectHeigth
            import random
            rGen = random.Random(leewayX)
            posX = rGen.uniform(0,leewayX)
            posY = rGen.uniform(0,leewayY)

            # create cmd
            cmdList.append( MoveObject( self.theLayout, objectID, posX, posY, systemObject ) )

        return cmdList

    def autoConnect( self, processObjectID, variableObjectID ):
        # source and target can be in a form of ( centerx, centery), too

        
        # get dimensions of object and x, y pos
        if type(processObjectID) in ( type(()), type([] ) ):
            aProObjectXCenter, aProObjectYCenter = processObjectID
        else:
            aProObject = self.theLayout.getObject( processObjectID )
            aProObjectWidth = aProObject.getProperty( OB_DIMENSION_X )
            aProObjectHeigth = aProObject.getProperty( OB_DIMENSION_Y )
            (aProObjectX1,aProObjectY1)=aProObject.getAbsolutePosition()
            aProObjectXCenter = aProObjectX1 + aProObjectWidth/2
            aProObjectYCenter = aProObjectY1 + aProObjectHeigth/2

        if type(variableObjectID) in ( type(()), type([] ) ):
            aVarObjectXCenter, aVarObjectYCenter = variableObjectID
        else:
            aVarObject = self.theLayout.getObject( variableObjectID )
            aVarObjectWidth = aVarObject.getProperty( OB_DIMENSION_X )
            aVarObjectHeigth = aVarObject.getProperty( OB_DIMENSION_Y )
            (aVarObjectX1,aVarObjectY1)=aVarObject.getAbsolutePosition()
            aVarObjectXCenter = aVarObjectX1 +aVarObjectWidth/2
            aVarObjectYCenter = aVarObjectY1 +aVarObjectHeigth/2

        #assign process ring n var ring
        return self.getRings( aProObjectXCenter, aProObjectYCenter, aVarObjectXCenter, aVarObjectYCenter )
        
    def createEntity( self, anEntityType, x, y ):
        # return command for entity and center
        # get parent system
        parentSystem = self.theLayout.getSystemAtXY( x, y )
        if parentSystem == None:
            return
        if anEntityType == ME_SYSTEM_TYPE:
            buttonType = PE_SYSTEM
        elif anEntityType == ME_PROCESS_TYPE:
            buttonType = PE_PROCESS
        elif anEntityType == ME_VARIABLE_TYPE:
            buttonType = PE_VARIABLE
        else:
            return None, 0, 0
        #get relcords
        command, width, height = parentSystem.createObject( x, y , buttonType )
        
        return  command,  width, height

    def autoShowObject( self, aFullID ):
        # return cmd or comes up with error message!
        pass
        
    def getRings( self, x0, y0, x1, y1 ):
        # return sourcering, targetring
        dy = y0-y1; dx = x1-x0;
        if dx == 0:
            dx = 0.0001
        if dy == 0:
            dy = 0.0001
        
        angle = math.atan( dy/dx )/math.pi*180
        if angle < 0:
            angle +=180
        if dy <0:
            angle += 180
        idx = int(angle/90 +0.5)%4
        return self.angleMatrix[idx]
        
    
        
