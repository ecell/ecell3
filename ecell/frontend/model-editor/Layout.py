#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
from Constants import *
from LayoutManager import *
from ModelStore import *
from PackingStrategy import *
from SystemObject import *
from ProcessObject import *
from VariableObject import *
from TextObject import *
from ConnectionObject import *
import math
try:
    import gnomecanvas 
except:
    import gnome.canvas as gnomecanvas


class Layout:

    def __init__( self, aLayoutManager, aName ):
        self.theCounter = 0
        self.theLayoutManager = aLayoutManager
        self.theLayoutBufferFactory = self.theLayoutManager.theLayoutBufferFactory
        self.thePackingStrategy = PackingStrategy( self )
        self.theLayoutBufferPaster = self.theLayoutManager.theLayoutBufferPaster
        self.theName = aName
        self.theObjectMap = {}
        self.thePropertyMap = {}
        
        default_scrollregion = [ -1000, -1000, 1000, 1000 ]
        self.orgScrollRegion = default_scrollregion

        self.thePropertyMap[ OB_DIMENSION_X ] = default_scrollregion[2] - default_scrollregion[0]
        self.thePropertyMap[ OB_DIMENSION_Y ] = default_scrollregion[3] - default_scrollregion[1]

        default_zoomratio = 1
        self.thePropertyMap[ LO_SCROLL_REGION ] = default_scrollregion
        self.thePropertyMap[ LO_ZOOM_RATIO ] = default_zoomratio
        self.theCanvas = None
        self.thePathwayEditor = None
        self.theSelectedObjectIDList = []
        

        # allways add root dir object
        anObjectID = self.getUniqueObjectID( ME_SYSTEM_TYPE )
        self.createObject( anObjectID, ME_SYSTEM_TYPE, ME_ROOTID, default_scrollregion[0], default_scrollregion[1], None )
        self.theRootObject = self.getObject( anObjectID )
        
        self.thePropertyMap[ LO_ROOT_SYSTEM ] = anObjectID
        #print str(anObjectID) + ' is objectID'
        #self.shift_press = False
        
    def update( self, aType = None, anID = None ):
        # i am not sure this is necessary
        pass

    def isShown( self ):
        return self.theCanvas != None

    def getName(self):
        return self.theName

    def getLayoutManager(self):
        return self.theLayoutManager

    def getPathwayEditor(self):
        return self.thePathwayEditor 

    def attachToCanvas( self, aCanvas ):
        self.theCanvas = aCanvas
        self.thePathwayEditor = self.theCanvas.getParentWindow()
        self.theCanvas.setLayout( self )
        # set canvas scroll region
        scrollRegion = self.getProperty( LO_SCROLL_REGION )
        self.theCanvas.setSize( scrollRegion )
        # set canvas ppu
        ppu = self.getProperty( LO_ZOOM_RATIO )
        self.theCanvas.setZoomRatio( ppu )
        self.theCanvas.scrollTo( scrollRegion[0], scrollRegion[1],'attach')
        # set canvas for objects and show objects


        self.__showObject( self.thePropertyMap[ LO_ROOT_SYSTEM ] )
        # then get all the connections, setcanvas, show

        for objectID in self.getObjectList(OB_TYPE_CONNECTION):
            anObject = self.theObjectMap[ objectID ]
            anObject.setCanvas( self.theCanvas )
            anObject.show()
        self.resumeSelection()
        
    def __showObject( self, anObjectID ):
        anObject = self.theObjectMap[ anObjectID ]
        anObject.setCanvas( self.theCanvas )
        anObject.show()
        if anObject.getProperty( OB_TYPE ) == OB_TYPE_SYSTEM:
            objectList = anObject.getObjectList()
            for anID in objectList:
                self.__showObject( anID )
        
        

    def detachFromCanvas( self ):
        
        # hide objects and setcanvas none
        for objectID in self.theObjectMap.keys():
            anObject = self.theObjectMap[ objectID ]
            anObject.setCanvas( None )
            anObject.hide()

        self.theCanvas = None
        


    def getCanvas( self ):
        return self.theCanvas
        

    def rename( self, newName ):
        self.theName = newName
        if self.thePathwayEditor!=None:
            self.thePathwayEditor.update()
        
        
    #########################################
    #           COMMANDS        #
    #########################################

    def __checkCounter ( self, objectID ):
        i=len(objectID) -1
        while not objectID[:i].isalpha():
            i-=1
        oid = long(objectID[i:])
        if oid> self.theCounter:
            self.theCounter = oid

    def createObject( self, objectID, objectType, aFullID=None, x=None, y=None, parentSystem = None  ):
        # object must be within a system except for textboxes 
        # parentSystem object cannot be None, just for root
        self.__checkCounter( objectID )
        if x == None and y == None:
            (x,y) = parentSystem.getEmptyPosition()

        if objectType == OB_TYPE_PROCESS:
            
            if parentSystem == None:
                parentSystem = self
            
            newObject = ProcessObject( self, objectID, aFullID, x, y, parentSystem )
        elif objectType == OB_TYPE_VARIABLE:
            if parentSystem == None:
                parentSystem = self
            newObject = VariableObject( self, objectID, aFullID, x, y, parentSystem )

        elif objectType == OB_TYPE_SYSTEM:
            if parentSystem == None:
                parentSystem = self
                
            newObject = SystemObject( self, objectID, aFullID, x, y, parentSystem )

        elif objectType == OB_TYPE_TEXT:
            if parentSystem == None:
                parentSystem = self
            newObject = TextObject( self, objectID, x, y, parentSystem )

        elif objectType == OB_TYPE_CONNECTION:
            raise "Connection object cannot be created via Layout.createObject"

        else:
            raise Exception("Object type %s does not exists"%objectType)
        
        self.theObjectMap[ objectID ] = newObject
        if self.theCanvas!=None:
            newObject.setCanvas( self.theCanvas )
            newObject.show()
            self.selectRequest( objectID )


    def deleteObject( self, anObjectID ):
        #unselect
        if  anObjectID in self.theSelectedObjectIDList:
            self.theSelectedObjectIDList.remove( anObjectID )
        anObject = self.getObject( anObjectID )
        aParent = anObject.getParent()
        anObject.destroy()
        if aParent != self:
            aParent.unregisterObject( anObjectID )
        self.theObjectMap.__delitem__( anObjectID )


    def getObjectList( self, anObjectType = None ):
        # returns IDs
        if anObjectType == None:
            return self.theObjectMap.keys()
        returnList = []
        for anID in self.theObjectMap.keys():
            anObject = self.theObjectMap[ anID ]
            if anObject.getProperty( OB_TYPE ) == anObjectType:
                returnList.append( anID )
        return returnList
            
            


    def getPropertyList( self ):
        return self.thePropertyMap.keys()
        
    
    def getProperty( self, aPropertyName ):
        if aPropertyName in self.thePropertyMap.keys():
            return self.thePropertyMap[aPropertyName]
        else:
            raise Exception("Unknown property %s for layout %s"%(self.theName, aPropertyName ) )

###################################################################
    def setProperty( self, aPropertyName, aValue ):
        self.thePropertyMap[aPropertyName] = aValue
        if aPropertyName==LO_SCROLL_REGION:
            scrollRegion=self.getProperty(LO_SCROLL_REGION)
            self.thePropertyMap[ OB_DIMENSION_X ]=scrollRegion[2]-scrollRegion[0]
            self.thePropertyMap[ OB_DIMENSION_Y ]=scrollRegion[3]-scrollRegion[1]
            if self.theCanvas!=None:
                self.theCanvas.setSize( scrollRegion )
            


    def getAbsoluteInsidePosition( self ):
        return ( 0, 0 )

    def moveObject(self, anObjectID, newX, newY, newParent ):
        # if newParent is None, means same system
        # currently doesnt accept reparentation!!!
        anObject = self.getObject( anObjectID )
        deltax = newX - anObject.getProperty( OB_POS_X ) 
        deltay = newY - anObject.getProperty( OB_POS_Y )
        anObject.move( deltax, deltay )

    def getObject( self, anObjectID ):
        # returns the object including connectionobject
        if anObjectID not in self.theObjectMap.keys():
            raise Exception("%s objectid not in layout %s"%(anObjectID, self.theName))
        return self.theObjectMap[ anObjectID ]


    def resizeObject( self, anObjectID, deltaTop, deltaBottom, deltaLeft, deltaRight ):
        # inward movement negative, outward positive
        anObject = self.getObject( anObjectID )
        anObject.resize( deltaTop, deltaBottom, deltaLeft, deltaRight )


    def createConnectionObject( self, anObjectID, aProcessObjectID = None, aVariableObjectID=None,  processRing=None, variableRing=None, direction = PROCESS_TO_VARIABLE, aVarrefName = None ):
        self.__checkCounter( anObjectID )
        # if processobjectid or variableobjectid is None -> no change on their part
        # if process or variableID is the same as connection objectid, means that it should be left unattached
        # direction is omitted
        newObject = ConnectionObject( self, anObjectID, aVariableObjectID, aProcessObjectID, variableRing, processRing, aVarrefName, self )
        
        self.theObjectMap[ anObjectID ] = newObject
        if self.theCanvas!=None:
            newObject.setCanvas( self.theCanvas )
            newObject.show()
            self.selectRequest( anObjectID )


    def redirectConnectionObject( self, anObjectID, newProcessObjectID, newVariableObjectID = None, processRing = None, variableRing = None, varrefName =None ):
        # if processobjectid or variableobjectid is None -> no change on their part
        # if process or variableID is the same as connection objectid, means that it should be left unattached
        conObject = self.getObject( anObjectID )
        conObject.redirectConnbyComm(newProcessObjectID,newVariableObjectID,processRing,variableRing,varrefName)


    #################################################
    #       USER INTERACTIONS       #
    #################################################

    def userMoveObject( self, ObjectID, deltaX, deltaY ):
        #to be called after user releases shape
        pass
        #return TRUE move accepted, FALSE move rejected


    def userCreateConnection( self, aProcessObjectID, startRing, targetx, targety ):
        pass
        #return TRUE if line accepted, FALSE if line rejected


    def autoAddSystem( self, aFullID ):
        pass


    def autoAddEntity( self, aFullID ):
        pass


    def autoConnect( self, aProcessFullID, aVariableFullID ):
        pass



    ####################################################
    #           OTHER              #
    ####################################################

    def getUniqueObjectID( self, anObjectType ):
        # objectID should be string
        self.theCounter += 1
        return anObjectType + str( self.theCounter )


    def getName( self ):
        return self.theName
        

    def graphUtils( self ):
        return self.theLayoutManager.theModelEditor.theGraphicalUtils


    def popupObjectEditor( self, anObjectID ):
        anObject = self.getObject( anObjectID )
        if anObject.getProperty(OB_TYPE) == OB_TYPE_CONNECTION:
            self.theLayoutManager.theModelEditor.createConnObjectEditorWindow(self.theName, anObjectID)
        else:
            if  anObject.getProperty(OB_HASFULLID): 
                self.theLayoutManager.theModelEditor.createObjectEditorWindow(self.theName, anObjectID)
            else:
                self.theLayoutManager.theModelEditor.printMessage("Sorry, not implemented!", ME_ERROR )


    def getPaletteButton( self ):
        return self.thePathwayEditor.getPaletteButton()


    def passCommand( self, aCommandList):
        self.theLayoutManager.theModelEditor.doCommandList( aCommandList)


    def registerObject( self, anObject ):
        pass


    

    def selectRequest( self, objectID , shift_press = False ):
        anObject = self.getObject( objectID )
        if len( self.theSelectedObjectIDList ) == 0:
            selectedType = None
        else:
            selectedType = self.getObject( self.theSelectedObjectIDList[0] ).getProperty( OB_TYPE )

        objectType = self.getObject(objectID ).getProperty(OB_TYPE )
            
        selectedObjectID = objectID
        multiSelectionAllowed = False
        if shift_press == False:
            #if shift_press == False: 
            for anID in self.theSelectedObjectIDList:
                self.getObject( anID ).unselected()
            self.theSelectedObjectIDList = []
        elif len( self.theSelectedObjectIDList ) > 0:
            if selectedType == OB_TYPE_CONNECTION:
                selectionSystemPath = None
            else:
                selectionSystemPath = self.getObject( self.theSelectedObjectIDList[0] ).getProperty( OB_FULLID ).split(':')[1]
            if objectType == OB_TYPE_CONNECTION:
                objectSystemPath  = None
            else:
                objectSystemPath = anObject.getProperty( OB_FULLID ).split(':')[1]
            multiSelectionAllowed = ( selectionSystemPath == objectSystemPath )
        else:
            multiSelectionAllowed = True

        if anObject.isSelected and shift_press:
            self.theSelectedObjectIDList.remove( objectID )
            anObject.unselected()
            if len(self.theSelectedObjectIDList) == 0:
                selectedObjectID = None
            else:
                selectedObjectID = self.theSelectedObjectIDList[-1]
        else:        
            if len(self.theSelectedObjectIDList)> 0:
                if shift_press and multiSelectionAllowed :
                    self.theSelectedObjectIDList += [objectID]
                    anObject.selected()
                else:
                    return
            else:
                self.theSelectedObjectIDList += [objectID]
                anObject.selected()
        
        if objectType == OB_TYPE_CONNECTION:
            self.theLayoutManager.theModelEditor.createConnObjectEditorWindow( self.theName, self.theSelectedObjectIDList )
        else:
            self.theLayoutManager.theModelEditor.createObjectEditorWindow( self.theName, self.theSelectedObjectIDList )
            
            
    def deleteEntities( self ):
        aModelEditor = self.theLayoutManager.theModelEditor
        aCommandList = []

        for anObjectID in self.theSelectedObjectIDList:
            anObject = self.getObject( anObjectID )
            if anObject.getProperty( OB_HASFULLID ):
                aFullID  = anObject.getProperty( OB_FULLID )
                aCommandList += [ DeleteEntityList( aModelEditor, [ aFullID ] ) ]

            elif anObject.getProperty( OB_TYPE )==OB_TYPE_CONNECTION:
                connObj = anObject
                varreffName = connObj.getProperty(CO_NAME)
                proID = connObj.getProperty(CO_PROCESS_ATTACHED)
                aProcessObject = self.getObject(proID)
                aProcessFullID = aProcessObject.getProperty( OB_FULLID )
                fullPN = aProcessFullID+':' +MS_PROCESS_VARREFLIST
                aVarReffList = copyValue( aModelEditor.theModelStore.getEntityProperty( fullPN ) )[:]

                for aVarref in aVarReffList:
                    if aVarref[ME_VARREF_NAME] == varreffName :
                        aVarReffList.remove( aVarref )
                        break
                aCommandList += [ ChangeEntityProperty( aModelEditor, fullPN, aVarReffList ) ]
        self.passCommand(  aCommandList  )

            
    def deleteSelected( self ):
        aCommandList = []
        for anObjectID in self.theSelectedObjectIDList:
            aCommandList += [ DeleteObject( self, anObjectID ) ]
        self.passCommand( aCommandList )

                
    def resumeSelection( self ):
        for anID in self.theSelectedObjectIDList:
            anObject = self.getObject( anID )
            anObject.selected()
        
    def checkConnection( self, x, y, checkFor ):
        objectIDList = self.getObjectList( checkFor )
        for anObjectID in objectIDList:
            anObject = self.theObjectMap[ anObjectID ]
            (objx1, objy1) = anObject.getAbsolutePosition()
            if x< objx1 - 3 or y < objy1 - 3:
                continue
            objx2 = objx1 + anObject.getProperty( OB_DIMENSION_X )
            objy2 = objy1 + anObject.getProperty( OB_DIMENSION_Y )

            if x > 3 + objx2 or y > 3 + objy2:
                continue
            rsize = anObject.getRingSize()
            chosenRing = None
            chosenDist = None
            for aRingName in [ RING_TOP, RING_BOTTOM, RING_LEFT, RING_RIGHT ]:
                (rx, ry) = anObject.getRingPosition( aRingName )
                distance = math.sqrt((rx-x)**2+(ry-y)**2)
                if chosenDist == None or chosenDist>distance:
                    chosenDist = distance
                    chosenRing = aRingName
                    
            return ( anObjectID, chosenRing )
        return ( None, None )
        
    def getSystemAtXY( self, x, y ):
        # return systemID at absolute position
        return self.theRootObject.getSystemAtXY( x, y)
