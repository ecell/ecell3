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

from ecell.ecssupport import *
        
def getNextInList(  aList, actualID ):
    if actualID == None:
        idx = -1
    else:
        idx = aList.index( actualID )
    
    if idx == len(aList) - 1:
        return None
    else:
        return aList[idx + 1]
        
def getPreviousInList(  aList, actualID ):
    if actualID == None:
        idx = len(aList)
    else:
        idx = aList.index( actualID )
    
    if idx == 0:
        return None
    else:
        return aList[idx - 1]

class ModelWalker:
    def __init__( self, aModel  ):
        """ aModel can be any that complies the ecell3 Simulator API"""

        self.theModel = aModel
        self.theTreeWalker = TreeWalker( aModel )
        self.reset()
        
        
        
    def moveTo( self, aFullID ):
        self.theActualID = list( aFullID )
        if self.theActualID[TYPE] == SYSTEM:
            self.theTreeWalker.moveTo( self.theActualID )
        else:
            self.theTreeWalker.moveTo( createFullIDFromSystemPath( self.theActualID[SYSTEMPATH]  ) )
        self.__getEntityLists( )
         
        
        
    def reset ( self ):
        self.moveTo( [ SYSTEM, '', '/' ] )

        
    def getCurrentFullID( self ):
        return self.theActualID


        
    def getNextFullID ( self ):
        """
        moves to next FullID
        gives None if reaches end of tree """
        previousID = self.theActualID[:]
        self.__moveToNextEntity()
        if createFullIDString( self.theActualID ) == 'System::/':
            self.moveTo( previousID)
            return None
        return self.getCurrentFullID()
        
        
        
    def getPreviousFullID( self ):
        """ 
        moves to Previous FullID
        gives None if reaches the root"""
        if createFullIDString( self.theActualID ) == 'System::/':
            return None
        self.__moveToPreviousEntity()
        return self.getCurrentFullID()
        
        
    def __moveToPreviousEntity( self ):
        # order :  System, Process, Variable , System up 
        if self.theActualID[TYPE] == SYSTEM:
            self.theActualID = self.theTreeWalker.getPreviousSystemFullID()[:]
            self.__getEntityLists()
            actualID = None
            systemPath = createSystemPathFromFullID( self.theActualID )

        else:
            actualID = self.theActualID[ID]
            systemPath = self.theActualID[SYSTEMPATH]
        
        if self.theActualID[TYPE] in [ PROCESS, SYSTEM ]:
            actualID = getPreviousInList( self.theProcessList, actualID  )
            if actualID != None:
                self.theActualID[ID] = actualID
                self.theActualID[SYSTEMPATH] = systemPath
                self.theActualID[TYPE] = PROCESS
                return
                
        actualID = getPreviousInList( self.theVariableList, actualID  )
        if actualID != None:
            self.theActualID[ID] = actualID
            self.theActualID[SYSTEMPATH] = systemPath
            self.theActualID[TYPE] = VARIABLE
            return
        # move to actual system
        self.theActualID = createFullIDFromSystemPath( systemPath )

    

    def __moveToNextEntity( self ):
        # order : Variable, Process, System, System inside
        if self.theActualID[TYPE] == SYSTEM:
            actualID = None
            systemPath = createSystemPathFromFullID( self.theActualID )
        else:
            actualID = self.theActualID[ID]
            systemPath = self.theActualID[SYSTEMPATH]
            
        # if variable getnext variable
        if self.theActualID[TYPE] in [VARIABLE, SYSTEM]:
            actualID = getNextInList( self.theVariableList, actualID  )
            if actualID != None:
                self.theActualID[ID] = actualID
                self.theActualID[SYSTEMPATH] = systemPath
                self.theActualID[TYPE] = VARIABLE
                return
                
        # getnext process

        actualID = getNextInList( self.theProcessList, actualID  )
        if actualID != None:
            self.theActualID[ID] = actualID
            self.theActualID[SYSTEMPATH] = systemPath
            self.theActualID[TYPE] = PROCESS
            return
        # move to next system
        self.theActualID = self.theTreeWalker.getNextSystemFullID()[:]
        self.__getEntityLists()
        

        
    def __getEntityLists( self ):
        if self.theActualID[TYPE] == SYSTEM:
            systemPath = createSystemPathFromFullID( self.theActualID )
        else:
            systemPath = self.theActualID[SYSTEMPATH]
        self.theProcessList = list( self.theModel.getEntityList( ENTITYTYPE_STRING_LIST[PROCESS], systemPath ) )
        self.theVariableList = list( self.theModel.getEntityList( ENTITYTYPE_STRING_LIST[VARIABLE], systemPath ) )

class TreeWalker:
    def __init__( self, aModel ):
        """ aModel can be any that cvomplies the ecell3 Simulator API"""
        self.theModel = aModel
        self.reset()

    def moveTo( self, aSystemFullID ):
        self.theActualID = aSystemFullID[:]
        self.__getLists()
        
    def reset( self ):
        self.moveTo( [ SYSTEM, '', '/' ] )
        
        
    def getNextSystemFullID( self ):
        #try to go down
        newID = getNextInList( self.theChildren, None )           
        if newID != None:

            self.theActualID = [ SYSTEM, createSystemPathFromFullID( self.theActualID ), newID ]
            self.__getLists()
            return self.theActualID

        # go up and forward recursively
        return self.__goForwardAndUp()
        
    
    def getPreviousSystemFullID( self ):

        # get backwards sibling and go to his bottom
        
        newID = getPreviousInList( self.theSiblings, self.theActualID[ID] )
        if newID != None:
            self.theActualID[ID] = newID
            self.__getLists()
            return self.__goDownAndForward()

        # go up and call self
        if len( self.theParentSiblings ) == 0:
            # already on top
            return self.theActualID

        self.theActualID = createFullIDFromSystemPath( self.theActualID[SYSTEMPATH]  )
        self.__getLists()

        return self.theActualID
        
        
    def __goForwardAndUp( self ):
        #try to go forwards
        newID = getNextInList( self.theSiblings, self.theActualID[ID] )
        if newID != None:
            self.theActualID[ID] = newID
            self.__getLists()
            return self.theActualID

        # try to move up
        if len( self.theParentSiblings ) == 0:
            # already on top
            return self.theActualID

        # move up
        self.theActualID = createFullIDFromSystemPath( self.theActualID[SYSTEMPATH]  )
        self.__getLists()

        # go up and forward recursively
        return self.__goForwardAndUp()
        
        
    def __goDownAndForward( self ):
        # go down
        newID = getNextInList( self.theChildren, None )           
        if newID != None:
            self.theActualID = [ SYSTEM, createSystemPathFromFullID( self.theActualID ), newID ]
            self.__getLists()

            #get last forwards
            
            newID = getPreviousInList( self.theSiblings, None )
            self.theActualID[ID] = newID
            self.__getLists()
            self.__goDownAndForward()

        return self.theActualID



    def __getLists( self ):
        if self.theActualID[SYSTEMPATH] == '':
            self.theParentSiblings = []
        elif self.theActualID[SYSTEMPATH] == '/':
            self.theParentSiblings = ['/']
        else:
            parentID = createFullIDFromSystemPath( self.theActualID[SYSTEMPATH] )
            self.theParentSiblings = list( self.theModel.getEntityList( ENTITYTYPE_STRING_LIST[SYSTEM], 
                                        parentID[SYSTEMPATH] ) )
        if self.theActualID[SYSTEMPATH] == '':
            self.theSiblings = ['/']
        else:
            self.theSiblings =  list( self.theModel.getEntityList( ENTITYTYPE_STRING_LIST[SYSTEM], 
                                            self.theActualID[SYSTEMPATH] ) )

        self.theChildren = list( self.theModel.getEntityList( ENTITYTYPE_STRING_LIST[SYSTEM],
                         createSystemPathFromFullID( self.theActualID ) ) )
        
    
