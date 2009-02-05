#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
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

from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.Command import *

__all__ = (
    'LayoutCommand',
    'CreateLayout',
    'DeleteLayout',
    'RenameLayout',
    'CloneLayout',
    'PasteLayout',
    'CreateObject',
    'DeleteObject',
    'ChangeLayoutProperty',
    'SetObjectProperty',
    'PasteObject',
    'UndeleteObject',
    'MoveObject',
    'ResizeObject',
    'CreateConnection',
    'RedirectConnection',
    )

class LayoutCommand( Command ):
    def __init__(self, aReceiver, *args):
        """
        convert Layout receiver to LayoutName
        """
        if aReceiver.__class__.__name__ == "Layout":
            self.theLayoutName = aReceiver.getName()
            self.theLayoutManager = aReceiver.getLayoutManager()
        else:
            self.theLayoutName = None
        Command.__init__( self, aReceiver, *args )
    
    def execute(self):
        """
        must convert  LayoutName into receiver
        """
        if self.theLayoutName != None:
            self.theReceiver = self.theLayoutManager.getLayout( self.theLayoutName )
        return Command.execute( self )

    def checkArgs( self ):
        if type (self.theReceiver) == type(self):
            if self.theReceiver.__class__.__name__ == self.RECEIVER:
                return True

        return False

class CreateLayout(LayoutCommand):
    """
    arg1: NAME
    """
    RECEIVER = 'LayoutManager'
    ARGS_NO = 2
    NAME = 0
    SHOW = 1

    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.theName = self.theArgs[ self.NAME ]
        self.isShow = self.theArgs[ self.SHOW ]
        #check if layout name exists
        if self.theReceiver.doesLayoutExist(self.theName):
            return False
        return True

    
    def do( self ):
        
        self.theReceiver.createLayout( self.theName)
        if self.isShow:
            self.theReceiver.showLayout(self.theName)
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = [ DeleteLayout( self.theReceiver, self.theName ) ]


    def getAffected( self ):
        return (self.RECEIVER, None )

class DeleteLayout(LayoutCommand):
    """
    arg1: NAME
    """
    RECEIVER = 'LayoutManager'
    ARGS_NO = 1
    NAME = 0


    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.theName = self.theArgs[ self.NAME ]

        #check if layout name exists
        if not self.theReceiver.doesLayoutExist(self.theName):
            return False
        return True

    
    def do( self ):
        # prepare copy of layout
        layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.theName )
        layoutBuffer.setUndoFlag( True )
        # check if layout was shown and set show flag in pastelayout command accorddingly
        aLayout = self.theReceiver.getLayout( self.theName )
        
        self.theReverseCommandList = [ PasteLayout( self.theReceiver, layoutBuffer, None, aLayout.isShown() ) ]
        self.theReceiver.deleteLayout( self.theName)
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = None


    def getAffected( self ):
        return (self.RECEIVER, None )

class RenameLayout(LayoutCommand):
    """
    arg1: NAME
    """
    RECEIVER = 'LayoutManager'
    ARGS_NO = 2
    OLDNAME = 0
    NEWNAME = 1

    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.newName = self.theArgs[ self.NEWNAME ]
        if not isIDEligible( self.newName ):
            return False

        self.oldName = self.theArgs[ self.OLDNAME ]
        #check if layout name exists
        if self.theReceiver.doesLayoutExist(self.newName):
            return False
        if not self.theReceiver.doesLayoutExist(self.oldName):
            return False
        return True


    def do( self ):
        self.theReceiver.renameLayout( self.oldName,self.newName)
        return True


    def createReverseCommand( self ):
        #self.theReverseCommandList = [ RenameLayout( self.theReceiver,  self.oldName, self.newName  ) ]
        self.theReverseCommandList = [ RenameLayout( self.theReceiver,  self.newName, self.oldName  ) ]


    def getAffected( self ):
        return (self.RECEIVER, None )

class CloneLayout(LayoutCommand):
    """
    arg1: TEMPLATE
    """
    RECEIVER = 'LayoutManager'
    ARGS_NO = 1
    TEMPLATE = 0

    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.theTemplate = self.theArgs[ self.TEMPLATE ]
        #check if layout name exists
        if not self.theReceiver.doesLayoutExist(self.theTemplate):
            return False
        return True


    def do(self):
        layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.theTemplate )
        layoutBuffer.setUndoFlag( True )
        newName = "copyOf" + self.theTemplate
        newName = self.theReceiver.getUniqueLayoutName( newName )
        self.theReceiver.theLayoutBufferPaster.pasteLayoutBuffer( layoutBuffer, newName )
        self.theReverseCommandList = [ DeleteLayout( self.theReceiver, newName ) ]
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = None

class PasteLayout(LayoutCommand):
    """
    arg1: layoutbuffer
    arg2: new name if no new name, submit None
    """
    RECEIVER = 'LayoutManager'
    ARGS_NO = 3
    BUFFER = 0
    NEWNAME = 1
    SHOW = 2

    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.theBuffer = self.theArgs[ self.BUFFER ]
        self.newName = self.theArgs[ self.NEWNAME ]
        if self.newName != None:
            if not isIDEligible( self.newName ):
                return False

        self.isShow = self.theArgs[ self.SHOW ]
        return True


    def do(self):
        overWrite = False
        if self.newName == None:
            self.newName = self.theBuffer.getName()

        if self.theReceiver.doesLayoutExist(self.newName):
            #if self.theReceiver.theModelEditor.printMessage( "Do you want to overwrite layout %s"%self.newName ) = ME_RESULT_OK:
            # get copy of layout
            layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.newName )
            layoutBuffer.setUndoFlag( True )
            #check if layougt was shown, and set flag in pastelayout command
            self.theReverseCommandList = [ PasteLayout( self.theReceiver, layoutBuffer, None, self.isShow ) ]
            self.theReceiver.deleteLayout( self.newName)
        else:
            self.theReverseCommandList = [ DeleteLayout( self.theReceiver, self.newName ) ]


        self.theReceiver.theLayoutBufferPaster.pasteLayoutBuffer( self.theBuffer, self.newName )
        if self.isShow:
            self.theReceiver.showLayout(self.newName)
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = None


    def getAffected( self ):
        return (self.RECEIVER, None )

class CreateObject(LayoutCommand):
    """
    args: objectid, type, fullid, x, y
    """
    RECEIVER = 'Layout'
    ARGS_NO = 6
    OBJECTID = 0
    TYPE = 1
    FULLID = 2
    X = 3
    Y = 4
    PARENT = 5

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.theType = self.theArgs[ self.TYPE ]
        self.theFullID = self.theArgs[ self.FULLID ]
        self.x = self.theArgs[ self.X ]
        self.y = self.theArgs[ self.Y ]
        self.theParentID = self.theArgs[ self.PARENT ].getID()
        #print self.theParentID
        #print self.theArgs
        return True


    def getID(self):
        return self.theArgs[ self.OBJECTID ]

    def do(self):
        theParent = self.theReceiver.getObject( self.theParentID )
        #print theParent
        self.theReceiver.createObject(self.objectID, self.theType, self.theFullID, self.x, self.y, theParent )
        return True


    def createReverseCommand( self ):
        
        self.theReverseCommandList = [ DeleteObject( self.theReceiver, self.objectID ) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class DeleteObject(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 1
    OBJECTID = 0

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        return True


    def do(self):
        objectBuffer = self.theReceiver.theLayoutBufferFactory.createObjectBuffer( self.theReceiver.getName(), self.objectID )
        anObject = self.theReceiver.getObject(self.objectID)
        aParent = anObject.getParent()
        if aParent.__class__.__name__ != 'Layout':
            aParentID = anObject.getParent().getID()
        else:
            aParentID ='System0'
        self.theReverseCommandList = [ UndeleteObject( self.theReceiver, objectBuffer, None, None, aParentID ) ]
        
        self.theReceiver.deleteObject(self.objectID)
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = None


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class ChangeLayoutProperty(LayoutCommand):
    """
    args:
    """
    RECEIVER = 'Layout'
    ARGS_NO=2
    PROPERTYNAME=0
    PROPERTYVALUE=1
    
    def checkArgs( self ):
        if not LayoutCommand.checkArgs(self):
            return False
        self.propertyName= self.theArgs[ self.PROPERTYNAME ]
        self.propertyValue= self.theArgs[ self.PROPERTYVALUE ]
        self.oldPropertyValue=self.theReceiver.getProperty(self.propertyName)
        return True

    def do( self ):
        self.theReceiver.setProperty(self.propertyName,self.propertyValue)
        return True

    def createReverseCommand( self ):
        self.theReverseCommandList=None
        if self.oldPropertyValue != None:
            revcom = ChangeLayoutProperty( self.theReceiver,  self.propertyName, self.oldPropertyValue )
            self.theReverseCommandList = [ revcom ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class SetObjectProperty(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 3
    OBJECTID = 0
    PROPERTYNAME = 1 # if None, get it from buffer
    NEWVALUE = 2 # if None get it from buffer


    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.propertyName = self.theArgs[ self.PROPERTYNAME ]
        self.newValue = self.theArgs[ self.NEWVALUE ]
        return True


    def do(self):
        # get object
        theObject = self.theReceiver.getObject( self.objectID )
        theObject.setProperty( self.propertyName, self.newValue )
        return True


    def createReverseCommand( self ):
        # store old value
        oldValue = copyValue( self.theReceiver.getObject(self.objectID).getProperty( self.propertyName ) )
        self.theReverseCommandList = [ SetObjectProperty( self.theReceiver, self.objectID, self.propertyName, oldValue ) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class PasteObject(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 4
    BUFFER = 0
    X = 1 # if None, get it from buffer
    Y = 2 # if None get it from buffer
    PARENTID = 3 # cannot be None


    def checkArgs( self ):
        # no argument check - suppose call is right
        self.theBuffer = self.theArgs[ self.BUFFER ]
        self.x = self.theArgs[ self.X ]
        self.y = self.theArgs[ self.Y ]
        self.theParentID = self.theArgs[ self.PARENTID ]
        return True


    def do(self):
        if self.theBuffer.__class__.__name__ == "MultiObjectBuffer":
            self.theReceiver.theLayoutBufferPaster.pasteMultiObjectBuffer( self.theReceiver, self.theBuffer, self.x, self.y, self.theParentID )
        else:
            self.theReceiver.theLayoutBufferPaster.pasteObjectBuffer( self.theReceiver, self.theBuffer, self.x, self.y, self.theParentID )
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = []
        if self.theBuffer.__class__.__name__ == "MultiObjectBuffer":
            for aSystemBufferName in self.theBuffer.getSystemObjectListBuffer().getObjectBufferList():
                aSystemBuffer = self.theBuffer.getSystemObjectListBuffer().getObjectBuffer( aSystemBufferName )
                self.__createReverseCommandForBuffer( aSystemBuffer )

            for aBufferName in self.theBuffer.getObjectListBuffer().getObjectBufferList():
                anObjectBuffer = self.theBuffer.getObjectListBuffer().getObjectBuffer( aBufferName )
                self.__createReverseCommandForBuffer( anObjectBuffer )
        else:
            self.__createReverseCommandForBuffer( self.theBuffer )

    def __createReverseCommandForBuffer( self, anObjectBuffer ):
        aType = anObjectBuffer.getProperty( OB_TYPE )

        if anObjectBuffer.getUndoFlag():
            newID = anObjectBuffer.getID()
        else:
            # get it from really pasted ones
            newID = self.theReceiver.getUniqueObjectID( aType )
            anObjectBuffer.setID( newID )
            anObjectBuffer.noNewID = True
        self.theReverseCommandList += [ DeleteObject( self.theReceiver,newID ) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class UndeleteObject(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 4
    BUFFER = 0
    X = 1 # if None, get it from buffer
    Y = 2 # if None get it from buffer
    PARENTID = 3 # cannot be None


    def checkArgs( self ):
        # no argument check - suppose call is right
        self.theBuffer = self.theArgs[ self.BUFFER ]
        self.x = self.theArgs[ self.X ]
        self.y = self.theArgs[ self.Y ]
        self.theParentID = self.theArgs[ self.PARENTID ]
        self.theBuffer.setUndoFlag ( True )
        return True


    def do(self):
        self.theReceiver.theLayoutBufferPaster.pasteObjectBuffer( self.theReceiver, self.theBuffer, self.x, self.y, self.theParentID )
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = [ ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class MoveObject(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 4
    OBJECTID = 0
    NEWX = 1
    NEWY = 2
    NEWPARENT = 3

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.newx = self.theArgs[ self.NEWX ]
        self.newy = self.theArgs[ self.NEWY ]
        #self.newParent = self.theArgs[ self.NEWPARENT ]
        self.newParent=None
        return True


    def do(self):
        a = self.theReceiver.getObject( self.objectID )
        self.theReceiver.moveObject( self.objectID, self.newx, self.newy, self.newParent )
        return True


    def createReverseCommand( self ):
        theObject = self.theReceiver.getObject( self.objectID )
        oldX = theObject.getProperty( OB_POS_X )
        oldY = theObject.getProperty( OB_POS_Y )
        self.theReverseCommandList = [ MoveObject( self.theReceiver, self.objectID, oldX, oldY ) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class ResizeObject(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 5
    OBJECTID = 0
    UP = 1 
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.up = self.theArgs[ self.UP ]
        self.down = self.theArgs[ self.DOWN ]
        self.left = self.theArgs[ self.LEFT ]
        self.right = self.theArgs[ self.RIGHT ]
        return True


    def do(self):
        self.theReceiver.resizeObject( self.objectID, self.up, self.down, self.left, self.right )
        return True


    def createReverseCommand( self ):
        antiUp = -self.up
        antiDown = -self.down
        antiLeft = -self.left
        antiRight = -self.right
        self.theReverseCommandList = [ ResizeObject( self.theReceiver, self.objectID, antiUp, antiDown, antiLeft, antiRight ) ]
        


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class CreateConnection(LayoutCommand):
    """
    args: objectid
    """
    RECEIVER = 'Layout'
    ARGS_NO = 7
    OBJECTID = 0
    PROCESSOBJECTID = 1
    VARIABLEOBJECTID = 2 
    PROCESSRING = 3
    VARIABLERING = 4
    DIRECTION = 5
    VARREFNAME = 6

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.processObjectID = self.theArgs[ self.PROCESSOBJECTID ]
        self.variableObjectID = self.theArgs[ self.VARIABLEOBJECTID ]
        self.processRing = self.theArgs[ self.PROCESSRING ]
        self.variableRing = self.theArgs[ self.VARIABLERING ]
        self.direction = self.theArgs[ self.DIRECTION ]
        self.varrefName = self.theArgs[ self.VARREFNAME ]
        return True


    def do(self):
        self.theReceiver.createConnectionObject( self.objectID, self.processObjectID, self.variableObjectID, self.processRing, self.variableRing, self.direction, self.varrefName )
        return True


    def createReverseCommand( self ):
        self.theReverseCommandList = [ DeleteObject( self.theReceiver, self.objectID ) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )

class RedirectConnection(LayoutCommand):
    """
    args: anObjectID, newProcessObjectID, newVariableObjectID = None, newRing = None 
       # arguments are None. means they dont change
    """
    RECEIVER = 'Layout'
    ARGS_NO = 6
    OBJECTID = 0
    NEWPROCESSOBJECTID = 1
    NEWVARIABLEOBJECTID = 2 # it is either a valid objectID or a pair of values [x,y] indicating the new endpoint
    NEWPROCESSRING = 3
    NEWVARIABLERING = 4
    NEWVARREFNAME = 5 # can be none

    def checkArgs( self ):
        # no argument check - suppose call is right
        self.objectID = self.theArgs[ self.OBJECTID ]
        self.newProcessObjectID = self.theArgs[ self.NEWPROCESSOBJECTID ]
        self.newVariableObjectID = self.theArgs[ self.NEWVARIABLEOBJECTID ]
        self.newProcessRing = self.theArgs[ self.NEWPROCESSRING ]
        self.newVariableRing = self.theArgs[ self.NEWVARIABLERING ]
        self.newVarrefName = self.theArgs[ self.NEWVARREFNAME ]
        return True
        

    def do(self):
        self.theReceiver.redirectConnectionObject( self.objectID, self.newProcessObjectID, self.newVariableObjectID, self.newProcessRing, self.newVariableRing,self.newVarrefName )
        return True


    def createReverseCommand( self ):
        theObject = self.theReceiver.getObject( self.objectID )
        if self.newProcessObjectID == None:
            oldProcessObjectID = None
            oldProcessRing = None
        else:
            oldProcessObjectID = theObject.getProperty( CO_PROCESS_ATTACHED )
            oldProcessRing = theObject.getProperty( CO_PROCESS_RING )
            

        if self.newVariableObjectID == None:
            oldVariableObjectID = None
            oldVariableRing = None
        else:
            
            oldVariableObjectID = theObject.getProperty( CO_VARIABLE_ATTACHED )
            if oldVariableObjectID == None:
                oldVariableObjectID = theObject.getProperty( CO_ENDPOINT2 )
            oldVariableRing = theObject.getProperty( CO_VARIABLE_RING )


        self.theReverseCommandList = [ RedirectConnection( self.theReceiver, self.objectID, oldProcessObjectID, oldVariableObjectID, oldProcessRing, oldVariableRing, self.newVarrefName) ]


    def getAffected( self ):
        return (self.RECEIVER, self.theReceiver )



