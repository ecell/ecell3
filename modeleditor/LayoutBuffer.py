from Buffer import *


class ObjectBuffer(Buffer):

    def __init__( self, anID ):
        self.theID = anID
        self.thePropertyBuffer = PropertyListBuffer()
        self.theEntityBuffer = None
        self.theParent = None
        self.undoFlag = False
        self.theConnectionBuffers = ObjectListBuffer()

    def getID( self ):
        return self.theID

    def getPropertyList( self ):
        return self.thePropertyBuffer.getPropertyList()

    def getProperty( self, aPropertyName ):
        return self.thePropertyBuffer.getProperty( aPropertyName )

    def getPropertyBuffer ( self ):
        return self.thePropertyBuffer

    def getEntityBuffer( self ):
        if self.theEntityBuffer == None:
            raise Exception( "Buffer of object %s doesn't have entitybuffer"%self.theID )
        return self.theEntityBuffer
        
    def setEntityBuffer( self, anEntityBuffer ):
        self.theEntityBuffer = anEntityBuffer

    def getConnectionList( self ):
        return self.theConnectionBuffers.getObjectBufferList()


    def addConnectionBuffer( self, objectBuffer ):
        self.theConnectionBuffers.addObjectBuffer( objectBuffer )

    def getConnectionBuffer( self, objectID ):
        return self.theConnectionBuffers.getObjectBuffer( objectID )

    def setUndoFlag( self, aValue):
        self.undoFlag = aValue
        self.theConnectionBuffers.setUndoFlag( aValue )

    def getUndoFlag( self ):
        return self.undoFlag



class ObjectListBuffer:

    def __init__( self ):
        self.theObjectList = {}


    def getObjectBufferList( self ):
        return self.theObjectList.keys()


    def getObjectBuffer( self, objectID ):
        return self.theObjectList[ objectID ]


    def addObjectBuffer( self, objectBuffer ):
        self.theObjectList[ objectBuffer.getID() ] = objectBuffer

    def setUndoFlag( self, aValue):
        for aBuffer in self.theObjectList.values():
            aBuffer.setUndoFlag( aValue )



class SystemObjectBuffer(ObjectBuffer):

    def __init__( self, anID ):
        ObjectBuffer.__init__( self, anID )
        self.theSingleObjectListBuffer = ObjectListBuffer()
        self.theSystemObjectListBuffer = ObjectListBuffer()

    def getSingleObjectListBuffer( self ):
        return self.theSingleObjectListBuffer


    def getSystemObjectListBuffer( self ):
        return self.theSystemObjectListBuffer


    def setUndoFlag( self, aValue):
        ObjectBuffer.setUndoFlag( self, aValue )
        self.theSingleObjectListBuffer.setUndoFlag( aValue )
        self.theSystemObjectListBuffer.setUndoFlag( aValue )
        

class MultiObjectBuffer( ObjectBuffer ):


    def __init__( self):
        ObjectBuffer.__init__(self, None )
        self.theObjectListBuffer = ObjectListBuffer()
        self.theSystemObjectListBuffer = ObjectListBuffer()
        self.theConnectionObjectListBuffer = ObjectListBuffer()


    def getSystemObjectListBuffer( self ):
        return self.theSystemObjectListBuffer


    def getObjectListBuffer( self ):
        return self.theTextObjectListBuffer


    def getObjectListBuffer( self ):
        return self.theObjectListBuffer




class LayoutBuffer( ObjectBuffer ):

    def __init__( self, aLayoutName ):
        ObjectBuffer.__init__(self, None)
        self.theName = aLayoutName
        self.theRootBuffer = None


    def getName( self ):
        #returns name of layout
        return self.theName

    def getRootBuffer( self ):
        return self.theRootBuffer
        
    def setRootBuffer ( self, anObjectBuffer ):
        self.theRootBuffer = anObjectBuffer
