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

from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.Layout import *
from ecell.ui.model_editor.LayoutBufferFactory import *
from ecell.ui.model_editor.PathwayCanvas import *

class LayoutManager:

    def __init__( self, aModelEditor):
        self.theModelEditor = aModelEditor
        self.theLayoutMap = {}
        self.theLayoutBufferFactory = LayoutBufferFactory( self.theModelEditor, self )
        self.theLayoutBufferPaster = LayoutBufferPaster( self.theModelEditor, self )
        self.theShowMap={}


    def createLayout( self, aLayoutName ):
        # create and show

        if aLayoutName in self.theLayoutMap.keys():
            raise Exception("Layout %s already exists!"%aLayoutName )
        newLayout = Layout ( self, aLayoutName )
        self.theLayoutMap[ aLayoutName ] = newLayout
        

    def deleteLayout( self, aLayoutName ):
        aLayout = self.theLayoutMap[ aLayoutName ]
        aCanvas=aLayout.getCanvas()
        aLayout.detachFromCanvas()
        del self.theLayoutMap[aLayoutName]
        if aCanvas!=None:
            editorWindow = aCanvas.getParentWindow()
            editorWindow.close()


    def showLayout( self, aLayoutName ):
        aLayout = self.theLayoutMap[ aLayoutName ]
        if aLayout.getCanvas() != None:
            return
        # create new pathwayeditor
        anEditorWindow = self.theModelEditor.createPathwayEditor( aLayout )



    def update( self, aType = None, anID = None ):
        # i am not sure this is necessary!!!
        self.theModelEditor.updateWindows()


    def getUniqueLayoutName( self, tryThisName = None ):
        if tryThisName == None:
            tryThisName = 'Layout'
        nameList = self.theLayoutMap.keys()

        counter = 0
        layoutName = tryThisName
        while layoutName in nameList:
            layoutName = tryThisName + str( counter )
            counter += 1
        return layoutName


    def getLayoutNameList( self ):
        return self.theLayoutMap.keys()


    def getLayout( self, aLayoutName ):
        return self.theLayoutMap[ aLayoutName ]
        

    def doesLayoutExist( self, aLayoutName ):
        return aLayoutName in self.theLayoutMap.keys()
    

    def renameLayout( self, oldLayoutName, newLayoutName ):
        aLayout = self.theLayoutMap[ oldLayoutName ]
        if self.doesLayoutExist( newLayoutName ):
            #raise Exception("%s layout already exists!"%newLayoutName )
            return False
        else:
            aLayout.rename( newLayoutName )
            self.theLayoutMap[ newLayoutName ] = aLayout
            self.theLayoutMap.__delitem__( oldLayoutName )
            return True

    def setLayoutProperty( self, aLayoutName, aPropertyName, aPropertyValue ):
        aLayout = self.theLayoutMap[ aLayoutName ]
        if self.doesLayoutExist( aLayoutName ):
            #raise Exception("%s layout already exists!"%newLayoutName )
            return False
        else:
            aLayout.setProperty( aPropertyName, aPropertyValue )
            return True
    
    def createObjectIterator( self ):
        return ObjectIterator( self)

class ObjectIterator:
    # cannot handle modifications to the layouts

    def __init__( self, aLayoutManager):
        self.theLayoutManager = aLayoutManager
        self.filterList = []
        self.reset()
        

    def reset( self ):
        self.layoutList = self.theLayoutManager.getLayoutNameList()
        self.objectList = []
        self.currentLayout = None
        self.currentObjectID = None
    

    def deleteFilters( self ):
        self.filterList = []
        self.reset()
    
    def filterByFullID( self, aFullID ):
        self.filterList.append( [ "FULLID", aFullID ] )
        
    def filterByProperty( self, aPropertyName, aPropertyValue ):
        self.filterList.append( [ "CUSTOMPROPERTY", aPropertyName, aPropertyValue ] )
    
    def filterByID( self, objectID ):
        self.filterList.append( "ID", objectID )


    def getNextObject( self ):
        # return first matching object self
        while self.__getNextObject() != None:
            theLayout = self.theLayoutManager.getLayout( self.currentLayout )
            theObject = theLayout.getObject( self.currentObjectID )
            if self.doesComply( theObject ):
                return theObject    
        return None
    
    
    def doesComply( self, anObject ):
        complied = 0
        propertyList = anObject.getPropertyList()
        for aFilter in self.filterList:
            if aFilter[0] == "FULLID" and OB_FULLID in propertyList:
                if aFilter[1] == anObject.getProperty( OB_FULLID ):
                    complied += 1
            elif aFilter[0] == "CUSTOMPROPERTY" and aFilter[1] in propertyList:
                if aFilter[2] == anObject.getProperty( aFilter[1] ):
                    complied += 1
            elif aFilter[0] == "ID" :
                if aFilter[1] == anObject.getID( ):
                    complied += 1
        return complied == len( self.filterList )


    
    def __getNextObject( self ):
        # if no objectlist or currentobject at the end of objectlist, get next layout
        if self.objectList != []:
            curpos = self.objectList.index( self.currentObjectID ) 
            if curpos != len( self.objectList ) - 1:
                curpos += 1
                self.currentObjectID = self.objectList[ curpos ]
                return self.currentObjectID
        self.__getNextLayout()

        if self.currentLayout != None:
            curpos = 0
            self.currentObjectID = self.objectList[ curpos ]
            return self.currentObjectID
        else:
            return None
                    

        
    def __getNextLayout( self ):
        # get a layout that contains at least one object
        # if last layout return None
        # fill in objectlist
        if self.layoutList == []:
            self.currentLayout = None
            return None
        if self.currentLayout == None:
            curpos = 0
        else:
            curpos = self.layoutList.index( self.currentLayout )
            curpos += 1
        while curpos < len( self.layoutList ):
            self.currentLayout = self.layoutList[curpos]
            layout = self.theLayoutManager.getLayout( self.currentLayout )
            self.objectList = layout.getObjectList()
            if self.objectList != []:
                return
            curpos += 1
        self.currentLayout = None
        return None
