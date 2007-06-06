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
from LayoutBuffer import *
from BufferFactory import *

class LayoutBufferFactory:

    def __init__( self, aModelEditor, aLayoutManager):
        self.theModelEditor = aModelEditor
        self.theLayoutManager = aLayoutManager


    def createBufferFromEml( self, anEml ):
        return None


    def createLayoutBuffer( self, aLayoutName ):
        aLayout = self.theLayoutManager.getLayout( aLayoutName )
        aLayoutBuffer = LayoutBuffer( aLayoutName )

        propertyBuffer = aLayoutBuffer.getPropertyBuffer()
        propertyList = aLayout.getPropertyList()
        for aProperty in propertyList:
            aValue = aLayout.getProperty( aProperty )
            propertyBuffer.createProperty( aProperty, aValue )


        # get all systemlist but copy only those whose parent is the layout self
        systemList = aLayout.getObjectList( OB_TYPE_SYSTEM )

        # add them to systemlist buffer
        for systemObjectID in systemList:
            systemObject = aLayout.getObject( systemObjectID )
            if systemObject.getParent() != aLayout or systemObject.getProperty( OB_TYPE ) != ME_SYSTEM_TYPE:
                continue
            systemBuffer = self.createObjectBuffer( aLayoutName, systemObjectID )
            aLayoutBuffer.setRootBuffer( systemBuffer )
            break

        # copy all connections
        #connectionList = aLayout.getObjectList( OB_TYPE_CONNECTION )
        #for connectionID in connectionList:
        #   connectionBuffer = self.createObjectBuffer( connectionID )
        #   aLayoutBuffer.getConnectionObjectListBuffer().addObjectBuffer( connectionBuffer )
        # copy all textboxes whose parent is layout
        #textList = aLayout.getObjectList( OB_TYPE_TEXT )
        #for textID in connectionList:
        #   textObject = aLayout.getObject( textID )
        #   if textObject.getParent() != aLayout:
        #       continue
        #   textBuffer = self.createObjectBuffer( textID )
        #   aLayoutBuffer.getTextObjectListBuffer().addObjectBuffer( textBuffer )
        return aLayoutBuffer


    def createMultiObjectBuffer( self, aLayoutName, anObjectIDList ):
        aMultiBuffer = MultiObjectBuffer()
        aLayout = self.theLayoutManager.getLayout( aLayoutName )
        topleftX = None
        topleftY = None
        bottomRightX = None
        bottomRightY = None
        for anObjectID in anObjectIDList:
            
            anObject = aLayout.getObject( anObjectID )
            left = anObject.getProperty( OB_POS_X )
            top = anObject.getProperty( OB_POS_Y )
            right = left + anObject.getProperty( OB_DIMENSION_X )
            bottom = top + anObject.getProperty( OB_DIMENSION_Y )
            if ( left <= topleftX or topleftX == None ): 
                topleftX = left
            if ( top <= topleftY or topleftY == None ):
                topleftY = top
            if ( right >= bottomRightX or bottomRightX == None ): 
                bottomRightX = right
            if ( bottom >= bottomRightY or bottomRightY == None ):
                bottomRightY = bottom

            aType = anObject.getProperty( OB_TYPE )
            aBuffer = self.createObjectBuffer( aLayoutName, anObjectID )
            if aType == OB_TYPE_SYSTEM:
                aMultiBuffer.getSystemObjectListBuffer().addObjectBuffer( aBuffer )
            else:
                aMultiBuffer.getObjectListBuffer().addObjectBuffer( aBuffer )

        aMultiBuffer.getPropertyBuffer().createProperty( OB_POS_X , topleftX )
        aMultiBuffer.getPropertyBuffer().createProperty( OB_POS_Y , topleftY )
        aMultiBuffer.getPropertyBuffer().createProperty( OB_DIMENSION_X ,   bottomRightX - topleftX )
        aMultiBuffer.getPropertyBuffer().createProperty( OB_DIMENSION_Y ,  bottomRightY - topleftY )
        return aMultiBuffer


    def createObjectBuffer( self, aLayoutName, anObjectID ):
        # but copy underlying entity
        aLayout = self.theLayoutManager.getLayout( aLayoutName )
        anObject = aLayout.getObject( anObjectID )
        aType = anObject.getProperty( OB_TYPE )
        if aType == OB_TYPE_SYSTEM:
            aBuffer = self.__createSystemObjectBuffer( anObject )
        else:
            aBuffer = self.__createObjectBuffer( anObject )
        return aBuffer
        
    
    def __createObjectBuffer ( self, anObject ):
        # TEXT, VARIABLE, PROCESS
        aType = anObject.getProperty( OB_TYPE )
        aParent = anObject.getParent()
        if aType == OB_TYPE_SYSTEM:
            aBuffer = SystemObjectBuffer( anObject.getID())
        else:
            aBuffer = ObjectBuffer( anObject.getID() )
        if aParent.__class__.__name__ == 'Layout':
            aBuffer.theParent = None
        else:
            aBuffer.theParent = aParent.getID()
        propertyBuffer = aBuffer.getPropertyBuffer()
        propertyList = anObject.getPropertyList()
        for aProperty in propertyList:
            aValue = anObject.getProperty( aProperty )
            if type( aValue ) == type( self ):
                continue
            # FIXME copy connections!
            if aProperty in [ VR_CONNECTIONLIST, PR_CONNECTIONLIST ]:
                aLayout = anObject.getLayout()
                for aConnID in aValue:
                    conObject = aLayout.getObject( aConnID )
                    aBuffer.addConnectionBuffer( self.__createObjectBuffer( conObject ) )
                continue
            propertyBuffer.createProperty( aProperty, aValue )
        if anObject.getProperty( OB_HASFULLID ):
            aFullID = anObject.getProperty( OB_FULLID )
            aBufferFactory = BufferFactory( self.theModelEditor.getModel() )
            anEntityBuffer = None
            if aType ==  OB_TYPE_SYSTEM and aFullID != MS_SYSTEM_ROOT:
                anEntityBuffer = aBufferFactory.createSystemListBuffer( [ aFullID ] )
        
            elif aType ==  OB_TYPE_VARIABLE:
                anEntityBuffer = aBufferFactory.createVariableListBuffer( [ aFullID ] )
        
            elif aType == OB_TYPE_PROCESS :
                anEntityBuffer = aBufferFactory.createProcessListBuffer( [ aFullID ] )

            aBuffer.setEntityBuffer( anEntityBuffer )
            
        return aBuffer


    def __createSystemObjectBuffer ( self, aSystemObject ):
        aBuffer = self.__createObjectBuffer( aSystemObject )
        childIDList = aSystemObject.getObjectList()
        aLayout = aSystemObject.getLayout()
        for aChildID in childIDList:
            childObject = aLayout.getObject( aChildID )
            aType = childObject.getProperty( OB_TYPE )
            if aType == OB_TYPE_SYSTEM:
                childSystemBuffer = self.__createSystemObjectBuffer( childObject )
                aBuffer.getSystemObjectListBuffer().addObjectBuffer( childSystemBuffer )
            elif aType != OB_TYPE_CONNECTION:
                childObjectBuffer = self.__createObjectBuffer( childObject )
                aBuffer.getSingleObjectListBuffer().addObjectBuffer( childObjectBuffer )
        return aBuffer


class LayoutBufferPaster:
    def __init__( self, aModelEditor, aLayoutManager):
        self.theModelEditor = aModelEditor
        self.theLayoutManager = aLayoutManager


    def pasteLayoutBuffer( self, aBuffer, newName = None ):
        # if newname is specified pastes with its name
        if newName == None:
            # get name from buffer
            newName = aBuffer.getName()
            if newName in self.theLayoutManager.getLayoutNameList():
                newName = "CopyOf" + newName
        # create layout
        self.theLayoutManager.createLayout( newName )
        
        aLayout = self.theLayoutManager.getLayout( newName )
        # paste properties
        aPropertyList = aBuffer.getPropertyList()
        for aProperty in aPropertyList:
            aValue = aBuffer.getProperty(aProperty)
            aLayout.setProperty( aProperty, aValue )
            
        translationList = {}
        

        # paste systemlistbuffer

        aSystemBuffer = aBuffer.getRootBuffer()
        oldSystemRootID = aLayout.getProperty( LO_ROOT_SYSTEM )
        aLayout.deleteObject( oldSystemRootID )

        anObject = self.pasteObjectBuffer(aLayout, aSystemBuffer, None, None, None, translationList )

        newRootID = anObject.getID()
        aLayout.setProperty( LO_ROOT_SYSTEM, newRootID )
        

        #self.__pasteObjectListBuffer( aBuffer.getTextObjectListBuffer(), aLayout, newName )
        #self.__pasteObjectListBuffer( aBuffer.getConnectionObjectListBuffer(), aLayout, newName )

    def pasteMultiObjectBuffer( self, aLayout, aBuffer, x, y, theParentID ):
        # get the topleft widget
        offsetX = aBuffer.getPropertyBuffer().getProperty( OB_POS_X ) - x
        offsetY = aBuffer.getPropertyBuffer().getProperty( OB_POS_Y ) - y
        translationList = {}
        pastedList = []
        for aSystemBufferName in aBuffer.getSystemObjectListBuffer().getObjectBufferList():
            aSystemBuffer = aBuffer.getSystemObjectListBuffer().getObjectBuffer( aSystemBufferName )
            posX = aSystemBuffer.getPropertyBuffer().getProperty( OB_POS_X ) - offsetX
            posY = aSystemBuffer.getPropertyBuffer().getProperty( OB_POS_Y ) - offsetY
            
            self.pasteObjectBuffer(aLayout, aSystemBuffer, posX  , posY, theParentID, translationList, pastedList )
        for aBufferName in aBuffer.getObjectListBuffer().getObjectBufferList():
            anObjectBuffer = aBuffer.getObjectListBuffer().getObjectBuffer( aBufferName )
            posX = anObjectBuffer.getPropertyBuffer().getProperty( OB_POS_X ) - offsetX
            posY = anObjectBuffer.getPropertyBuffer().getProperty( OB_POS_Y ) - offsetY

            self.pasteObjectBuffer(aLayout, anObjectBuffer, posX, posY, theParentID, translationList, pastedList )


    def pasteObjectBuffer( self, aLayout, aBuffer, x = None, y = None, theParentID = None, translationList = None, pastedList = None):
        if translationList == None:
            translationList = {}
        # dont create connections and entities
        
        aType = aBuffer.getProperty( OB_TYPE )
        
        if aBuffer.getUndoFlag():
            theParentID = aBuffer.theParent
            if theParentID != None:
                theParent = aLayout.getObject( theParentID )
            else:
                theParent = None

        if aType in ( OB_TYPE_VARIABLE, OB_TYPE_PROCESS, OB_TYPE_TEXT, OB_TYPE_SYSTEM):
            if theParentID != None:
                theParent = aLayout.getObject( theParentID )
            else:
                theParent = None

            anObject = self.__pasteObjectBuffer( aLayout, aBuffer, x, y, theParent, translationList )
            self.__pasteObjectConnections( aLayout, aBuffer, theParent, translationList, pastedList )
            return anObject
        else:
            self.__pasteConnectionObjectBuffer( aLayout, aBuffer, translationList )
            


    
    def __pasteObjectListBuffer( self, listBuffer, aParent, aLayoutName, translationList = None ):
        objectList = listBuffer.getObjectBufferList()

        aLayout = self.theLayoutManager.getLayout( aLayoutName )
        for anObjectID in objectList:
            aBuffer = listBuffer.getObjectBuffer( anObjectID )
            self.__pasteObjectBuffer( aLayout, aBuffer, None, None, aParent, translationList )
        

        

    def __pasteObjectBuffer( self, aLayout, aBuffer, x, y, theParent, translationList ):
    
        aType = aBuffer.getProperty( OB_TYPE )


        if x == None:
            x = aBuffer.getProperty( OB_POS_X )
        if y == None:
            y = aBuffer.getProperty( OB_POS_Y )
        aFullID = None
        if aBuffer.getProperty( OB_HASFULLID ):
            anObjName = aBuffer.getProperty( OB_FULLID ).split(':')[2]
            if theParent !=None:
                aSystemPath = convertSysIDToSysPath(theParent.getProperty(OB_FULLID))
            else:
                aSystemPath = ''

            aFullID = ":".join( [aType, aSystemPath, anObjName ] )

        # get ID
        oldID = aBuffer.getID()
        if aBuffer.getUndoFlag() or aBuffer.noNewID:
            newID = oldID
        else:
            newID = aLayout.getUniqueObjectID( aType )

        translationList[oldID] = newID
        aLayout.createObject( newID, aType, aFullID, x, y, theParent  )
        anObject = aLayout.getObject( newID )
        propertyList = aBuffer.getPropertyList()
        for aProperty in propertyList:
            if aProperty not in (OB_POS_X , OB_POS_Y, OB_FULLID):
                aValue = aBuffer.getProperty( aProperty )
                anObject.setProperty( aProperty, aValue )
            anObject.setProperty(OB_POS_X, x )
            anObject.setProperty(OB_POS_Y, y )
            anObject.setProperty(OB_FULLID, aFullID )
            
            
        if aType == OB_TYPE_SYSTEM:
            aLayoutName = aLayout.getName()
            self.__pasteObjectListBuffer(  aBuffer.getSingleObjectListBuffer(), anObject, aLayoutName, translationList )
            self.__pasteObjectListBuffer(  aBuffer.getSystemObjectListBuffer(), anObject, aLayoutName, translationList )

        return anObject # for systems

        

#   def __pasteSystemObjectBuffer( self,  aLayout, aType, aBuffer, x, y, theParent, translationList ):
#       aSystem = self.__pasteObjectBuffer( aLayout, aType, aBuffer, x, y, theParent, translationList )
#       aLayoutName = aLayout.getName( )
#       self.__pasteObjectListBuffer(  aBuffer.getSingleObjectListBuffer(), aSystem, aLayoutName, translationList )
#       self.__pasteObjectListBuffer(  aBuffer.getSystemObjectListBuffer(), aSystem, aLayoutName, translationList )
        
        
#   def __pasteConnectionObjectBuffer( self, aLayout, aBuffer ):
#       # makes sense only when copying layouts
#       anID = aBuffer.getID()
#       if anID in aLayout.getObjectList( OB_TYPE_CONNECTION ):
#           anID = aLayout.getUniqueObjectID()
#       processID = aBuffer.getProperty( CO_PROCESS_ATTACHED ).getID()
#       variableID = aBuffer.getProperty( CO_VARIABLE_ATTACHED ).getID()
#       processRing = aBuffer.getProperty( CO_PROCESS_RING )
#       variableRing = aBuffer.getProperty( CO_VARIABLE_RING )
#       aVarrefName = aBuffer.getProperty( CO_NAME )
#       direction = aBuffer.getProperty( CO_DIRECTION )
#       aLayout.createConnectionObject( anID, processID , variableID,  processRing, variableRing, direction, aVarrefName )
#       anObject = aLayout.getObject( anID )
#       propertyList = aBuffer.getPropertyList()
#       for aProperty in propertyList:
#           aValue = aBuffer.getProperty( aProperty )
#           anObject.setProperty( aProperty, aValue )


    def __pasteObjectConnections( self, aLayout, aBuffer, theParent, translationList, pastedList):
        if pastedList == None:
            pastedList = []
        
        # FIXME!!!
        # if system:
        aType = aBuffer.getProperty( OB_TYPE )
        if aType == OB_TYPE_SYSTEM:
            # get singleobjectlist cycle through it, call self:
            for anID in aBuffer.getSingleObjectListBuffer().getObjectBufferList():
                anObjectBuffer = aBuffer.getSingleObjectListBuffer().getObjectBuffer(anID)
                self.__pasteObjectConnections( aLayout, anObjectBuffer, theParent, translationList, pastedList)
            # get systemobjectlist, cycle through it, call self
            for anID in aBuffer.getSystemObjectListBuffer().getObjectBufferList():
                anObjectBuffer = aBuffer.getSystemObjectListBuffer().getObjectBuffer(anID)
                self.__pasteObjectConnections( aLayout, anObjectBuffer, theParent, translationList, pastedList)

        elif aType in [ OB_TYPE_VARIABLE, OB_TYPE_PROCESS]:
        # if process or variable
            # get connectionlist
            
            aBufferList = aBuffer.getConnectionList()
            for anID in aBufferList:
                # get connectionID
                
                if anID in pastedList:
                    continue
                # if on pastedList skip
                conBuffer = aBuffer.getConnectionBuffer( anID )
                # get process and variableid
                newID = self.__pasteConnectionObjectBuffer( aLayout, conBuffer, translationList )
                if newID != None:
                    pastedList.append( anID )


    def __pasteConnectionObjectBuffer( self, aLayout, conBuffer, translationList ):
        anID = conBuffer.getID()
        
        processID = conBuffer.getProperty( CO_PROCESS_ATTACHED )

        if processID not in translationList.keys():
            if not conBuffer.getUndoFlag():
                return None
            else:
                newProcessID = processID
        else:
            newProcessID = translationList[ processID ]
        aProcessObj = aLayout.getObject(newProcessID)

        # if not on translationlist skip
        variableID = conBuffer.getProperty( CO_VARIABLE_ATTACHED )
        if variableID  != None:
            if variableID not in translationList.keys():
                if not conBuffer.getUndoFlag():
                    return None
                else:
                    newVariableID = variableID
            else:
                newVariableID = translationList[variableID]
            varObj = aLayout.getObject(newVariableID)
        else:
            newVariableID = conBuffer.getProperty( CO_ENDPOINT2 )
            
        variableRing = conBuffer.getProperty( CO_VARIABLE_RING )
        processRing = conBuffer.getProperty( CO_PROCESS_RING )
        aVarrefName = conBuffer.getProperty( CO_NAME )
        # if not on translationlist and not none skip
        # if not undoFlag create new ID
        if not conBuffer.getUndoFlag():
            newID = aLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
        else:
            newID = anID
        # paste connection buffer
        # create new connection
        #aVarobj = aLayout.getObject( newVariableID )
        aLayout.createConnectionObject( newID, newProcessID, newVariableID,  processRing, variableRing, None, aVarrefName )
#        return newID

        conObject = aLayout.getObject( newID )
        # set properties
        propertyList = conBuffer.getPropertyList()
        for aProperty in propertyList:
            # endpoints should not be overwritten here
            if aProperty not in (CO_ENDPOINT1, CO_ENDPOINT2, CO_PROCESS_ATTACHED, CO_VARIABLE_ATTACHED):

                aValue = conBuffer.getProperty( aProperty )
                conObject.setProperty( aProperty, aValue )

        return newID
