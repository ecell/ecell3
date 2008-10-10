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

import os
import re
import numpy as nu
import math

from ModelEditor import *
from Utils import *
from Constants import *
from LayoutCommand import *
from ConfirmWindow import *

POINTS_PER_INCH = 36
FACTOR = 1.5

class AutoLayout:
    
    def __init__( self, aModelEditor, layoutName, listOfSelectionFullIds, entityList = False ):
        self.theModelEditor = aModelEditor
        self.theModelStore = self.theModelEditor.theModelStore

        self.theLayoutManager = self.theModelEditor.theLayoutManager
        self.theLayoutName = layoutName
        self.theLayoutManager.createLayout( layoutName)
        self.theLayout = self.theLayoutManager.getLayout( layoutName )
        self.theLayoutBufferFactory = self.theLayoutManager.theLayoutBufferFactory
        self.edgeRE = re.compile('--|->')
        self.secondaryTree = {ME_ROOTID:[{}]}
        self.allSysID = {ME_ROOTID: self.secondaryTree[ME_ROOTID][0]}
        self.variableReferenceList = [] #processID, variableID, 
        self.theUniqueCounter=0
        self.translationList = {}
        self.attributeMap={}
        self.edgeAttributeMap={}
        self.objectIDMap = {}
        self.theShapePluginManager = self.theModelEditor.getShapePluginManager()
        self.theGraphUtils = self.theModelEditor.theGraphicalUtils
        if entityList:
            listOfSelectionFullIds = self.__collectOtherEntities( listOfSelectionFullIds )
        self.generateAutoLayout(listOfSelectionFullIds, layoutName)


    def __collectOtherEntities( self, aList ):
        # homogenous entity list
        # get related entities
        newList = []
        aType = getFullIDType( aList[0] )
        for anEntity in aList:
            if aType == ME_PROCESS_TYPE:
                varrefList = self.theModelStore.getEntityProperty( anEntity + ":" + ME_PROCESS_VARREFLIST )
                for varref in varrefList:
                    aVarref =  getAbsoluteReference( anEntity, varref[ME_VARREF_FULLID] )
                    variableID = ":".join( [ ME_VARIABLE_TYPE ] + aVarref.split(":")[1:] )
                    if variableID not in newList:
                        newList.append( variableID )
            else:
                processList = self.theModelStore.getEntityProperty( anEntity + ":" + ME_VARIABLE_PROCESSLIST )
                for processID in processList:
                    if processID not in newList:
                        newList.append( processID )
        return newList + aList

    def __getSecondarySelection( self, primarySelection ):
        if ME_ROOTID in primarySelection:
            self.secondaryTree[ME_ROOTID] = [ self.__addSubSystems( ME_ROOTID ) ]
            self.allSysID[ME_ROOTID] = self.secondaryTree[ME_ROOTID][0]
            return


        
        for aFullID in primarySelection:
            parentDict = self.__addSuperSystems( aFullID )
            if aFullID in parentDict.keys():
                continue
            aType = getFullIDType( aFullID )
            if aType == ME_SYSTEM_TYPE:
                parentDict[aFullID] = [self.__addSubSystems( aFullID )]
                self.allSysID[aFullID] = parentDict[aFullID][0]
            else:
                parentDict[aFullID] = [None]
                
        
        
    def __addSuperSystems( self, anEntityFullID ):
        #returns pointer to addedsupersystems dictionary
        parentSystemID = getParentSystemOfFullID( anEntityFullID )
        parentDict = self.__getSystemDict( parentSystemID )
        if  parentDict == None:
            parentOfParentDict = self.__addSuperSystems( parentSystemID )
            parentDict = {}
            parentOfParentDict[parentSystemID] = [parentDict]
            self.allSysID[parentSystemID] = parentDict
        return parentDict
        
    def __getSystemDict( self, aSystemFullID ):
        if aSystemFullID in self.allSysID.keys():
            return self.allSysID[ aSystemFullID ]
        return None

    def __addSubSystems( self, aSystemFullID ):
        #returns dictionary containing subsystems
        aDict = {}
        systemPath = convertSysIDToSysPath( aSystemFullID )
        # get processes
        processList = self.theModelStore.getEntityList(ME_PROCESS_TYPE, systemPath )
        processIDList = map( lambda x : ':'.join([ME_PROCESS_TYPE, systemPath, x]), processList )
        for processID in processIDList:
            aDict[processID] = [None]
        # get variables
        variableList = self.theModelStore.getEntityList(ME_VARIABLE_TYPE, systemPath )
        if "SIZE" in variableList:
            variableList.remove( "SIZE" )
        variableIDList = map( lambda x : ':'.join([ME_VARIABLE_TYPE, systemPath, x]), variableList )
        for variableID in variableIDList:
            aDict[variableID] = [None]
        #this is a temporary solution until clustering is done!
        return aDict
        # get systems
        systemList = self.theModelStore.getEntityList(ME_SYSTEM_TYPE, systemPath )
        systemIDList = map( lambda x : ':'.join([ME_SYSTEM_TYPE, systemPath, x]), systemList )
        for systemID in systemIDList:
            aDict[systemID] = [self.__addSubSystems( systemID)]
            self.allSysID[systemID] = aDict[systemID][0]
        return aDict
            
        
        
    def __getVariableReferenceList( self ):
        for aDict in self.allSysID.values():
            for fullID in aDict.keys():
                if fullID.startswith( ME_PROCESS_TYPE ):
                    varrefList = self.theModelStore.getEntityProperty( fullID+':VariableReferenceList' )
                    for aVarref in varrefList:
                        aName = aVarref[0]
                        aVariableID = getAbsoluteReference( fullID, aVarref[1] )
                        vl = aVariableID.split(':')
                        aVariableID = ':'.join( [ME_VARIABLE_TYPE, vl[1], vl[2]] )
                        aCoef = aVarref[2]
                        varParentDict = self.__getSystemDict( getParentSystemOfFullID( aVariableID ) )
                        if varParentDict == None:
                            continue
                        if aVariableID not in varParentDict.keys():
                            continue
                       
                        self.variableReferenceList.append(  [fullID, aVariableID, aName, aCoef] )

             
    def __getUniqueID( self ):
        self.theUniqueCounter += 1
        return "ID"+str(self.theUniqueCounter)
       
    def __parseEntities( self, systemID ):

        if systemID == ME_ROOTID:
            ID = self.__getUniqueID()
            text='digraph '+ID+' {  graph [ label="'+systemID+'", labelloc="t" ] \n'
        else:
            ID = 'cluster'+self.__getUniqueID()
            text='subgraph ' + ID + ' {  graph [ label="'+systemID+'"] \n'
        self.translationList[ID] = systemID
        self.translationList[systemID] = ID
        aDict = self.allSysID[systemID]
        for fullID in aDict.keys():
            if fullID.startswith(ME_SYSTEM_TYPE ):
                text+=self.__parseEntities( fullID )
            else:
                ID = self.__getUniqueID()
                label = fullID.split(':')[2]
		label = label.replace('{','_')
                text += ID + '  [label="' + label + '"];\n'
                self.translationList[ID] = fullID
                self.translationList[fullID]=ID
        if systemID == ME_ROOTID:
            text += self.__parseVarrefs() + '\n}\n'
        else:
            text+='};\n'
        return text

    def __processLabel( self, aLabel ):
        return re.sub(r'[\[\];{}]','_',aLabel)


    def __parseVarrefs(self):
        text=""
        inList = {}
        outList = {}
        
        for aVarrefList in self.variableReferenceList:
            pnodeID = self.translationList[aVarrefList[0]]
            vnodeID = self.translationList[aVarrefList[1]]
            
            label = aVarrefList[2]
            coef = int(aVarrefList[3])
            if coef >0:
                edgeOp = pnodeID + "->" + vnodeID
                if inList.has_key(pnodeID):
                    inList[pnodeID] += [ vnodeID ]
                else:
                    inList[pnodeID] = [ vnodeID ]
            elif coef <0:
                edgeOp = vnodeID + "->" + pnodeID
                if outList.has_key(pnodeID):
                    outList[pnodeID] += [ vnodeID ]
                else:
                    outList[pnodeID] = [ vnodeID ]
            else:
                edgeOp = vnodeID + "->" + pnodeID
            text+= edgeOp +  ';\n'
#        for pnodeID in outList.keys():
#            if inList.has_key( pnodeID ):
#                for outvnodeid in outList[pnodeID]:
#                    for invnodeid in inList[pnodeID]:
#                        if outvnodeid != invnodeid:
#                            text += outvnodeid + "->" + invnodeid + " [ label = '_' ];\n"
        return text
        
    def __parseGraph( self, anID, nextLinePos ):
        while nextLinePos < len(self.theText ):
            aLine = self.theText[nextLinePos].strip()
            # decide whether, attrlist, graph, edge, node
            if aLine.startswith( 'subgraph'):
                # subgraph
                subID = re.findall('\w+[^{^ ]',aLine)[1]
                nextLinePos = self.__parseGraph(subID, nextLinePos + 1)
            elif aLine.startswith( 'graph'):
                # graph attributes
                self.__parseAttributes( anID, aLine )
            elif self.edgeRE.search( aLine )!=None:
                # edge
                self.__parseEdge( aLine )
            elif aLine.startswith( "ID" ):
                # node
                self.__parseNode( aLine )
            elif aLine.endswith("}"):
                return nextLinePos 
            nextLinePos += 1
            


    def __parseNode( self, aLine ):
        anID = re.findall(r'ID\d+',aLine)[0]
        self.__parseAttributes( anID, aLine )

    def __parseEdge( self, aLine ):
        anID = ":".join( re.findall(r'ID\d+', aLine ) )
        self.__parseAttributes( anID, aLine )
        
    def __parseAttributes( self, anID, aLine ):
        attributeList = re.findall(r'\w+=\"[e\d+\,\ \.]+\"',aLine)
        aMap = {}
        for attribute in attributeList:
            name, value = attribute.split('=')
            aMap[name] = map( float, re.findall(r'\d+\.?\d*', value ) )
        attributeList = re.findall(r'\w+=\"?[^ ^,]+\"?',aLine)
        aLabel = ""
        for attribute in attributeList:
            name, value = attribute.split('=')
            if name == "label":
                aLabel = value.strip('"')
                
        
        if not anID.__contains__( ":"):
            attMap = self.attributeMap
        else:
            attMap = self.edgeAttributeMap
            anID+=":"+aLabel

        if anID in attMap.keys():
            for anItem in aMap.items():
                attMap[anID][anItem[0]] = anItem[1]
        else:
            attMap[anID] = aMap

        
        
    def generateAutoLayout( self, primarySelection, layoutName ):

        # get secondary selection
        self.__getSecondarySelection( primarySelection )
        self.__getVariableReferenceList( )
	# parse into dot
        text = self.__parseEntities( ME_ROOTID)
        # write to file
        outputFileName =os.getcwd() + os.sep + "dotout"+str(os.getpid() )
        inputFileName = os.getcwd() + os.sep + "dotin" + str( os.getpid() ) 
        fd = open(outputFileName,'w')
        fd.write(text)
        fd.close()
        # process file
        os.system('neato -Tdot "' + outputFileName + '" -o"' + inputFileName + '"')
        #load input file
        try:
            fd = open( inputFileName, 'r')
        except:
            ConfirmWindow( 0, "Auto visualization function needs Graphviz installed.\n Please download and install this free software from http://www.graphviz.org/Download.php.\n Both windows and linux binaries are available.", "")
            return

        intext = " ".join(fd.readlines() )
        fd.close()
        intext = re.sub(r'[\\\n\t]','',intext)
        intext = re.sub('{','{\n',intext)
        intext = re.sub('}','}\n',intext)
        intext = re.sub(';',';\n',intext)
        self.theText = intext.split('\n')
        del intext
        
        
        keyWord, parentID = self.theText[0].strip().split(' ')[0:2]
        if keyWord != "digraph" and parentID!= self.translationList[ME_ROOTID]:
            ConfirmWindow(0, "There was an error processing dot file.\n No layout is created")
            return

        self.__parseGraph( parentID, 1 )
        

        # create systems
        self.__createBB ( ME_ROOTID )
        self.__createSystems(  ME_ROOTID )
        
        self.__createConnections()
        os.unlink ( inputFileName )
        os.unlink( outputFileName )        
        
        
        # create varrefs


        #layoutBuffer = self.theLayoutBufferFactory.createLayoutBuffer(self.theLayoutName)
        #self.theLayoutManager.deleteLayout(self.theLayoutName)
        #aCommand = PasteLayout( self.theLayoutManager, layoutBuffer , self.theLayoutName, True )
        #self.theModelEditor.doCommandList( [ aCommand ] )
        self.theLayoutManager.showLayout( self.theLayoutName )



    def __createSystems( self, aFullID, parentObjectID = None ):
        # get coordinates
        dotID = self.translationList[ aFullID ]
        bb = self.attributeMap[dotID]["bb"]

        width = bb[2]-bb[0]
        height = bb[3]-bb[1]
        # create system
        if aFullID == ME_ROOTID:

            theObjectID = self.theLayout.getProperty(LO_ROOT_SYSTEM)
            theSystemObject = self.theLayout.getObject(theObjectID)

            layoutbb = self.theLayout.getProperty( LO_SCROLL_REGION )
            deltaup = layoutbb[1] - bb[1] 
            deltaleft =  layoutbb[0] - bb[0] 
            deltaright = bb[2] -layoutbb[2] 
            deltadown =  bb[3] - layoutbb[3]

            theSystemObject.resize( deltaup, deltadown, deltaleft, deltaright )
            theSystemObject.adjustLayoutCanvas( deltaup, deltadown, deltaleft, deltaright )

            scrollRegion = self.theLayout.getProperty(LO_SCROLL_REGION )
            self.origo = [scrollRegion[0], scrollRegion[3] ]
            
            
            
        else:
            parentSystem = self.theLayout.getObject( parentObjectID )
            theObjectID = self.theLayout.getUniqueObjectID( OB_TYPE_SYSTEM )
            relx, rely = self.__convertAbsToRel( bb[0], bb[1], parentSystem )
            self.theLayout.createObject( theObjectID, OB_TYPE_SYSTEM, aFullID, relx, rely, parentSystem )
            #resize
            theSystemObject = self.theLayout.getObject( theObjectID )
            deltaright = - theSystemObject.getProperty( OB_DIMENSION_X ) + width
            deltadown = - theSystemObject.getProperty( OB_DIMENSION_Y ) + height
            theSystemObject.resize( 0, deltadown, 0, deltaright )
        
        aDict = self.__getSystemDict( aFullID )
        
        for fullID in aDict.keys():
            aType = getFullIDType( fullID )
            if aType == ME_SYSTEM_TYPE:
                self.__createSystems( fullID, theObjectID )
            elif aType in [ ME_VARIABLE_TYPE, ME_PROCESS_TYPE ]:
                entityObjectID = self.theLayout.getUniqueObjectID( aType )
                entityDotID = self.translationList[ fullID ]
                bb = self.attributeMap[entityDotID]["bb"]
                relx, rely = self.__convertAbsToRel( bb[0], bb[1], theSystemObject )
                self.theLayout.createObject( entityObjectID, aType, fullID, relx, rely, theSystemObject )
                self.objectIDMap[ fullID ] = entityObjectID


    def __createBB( self, aFullID, parentObjectID = None ):
        # get coordinates
        dotID = self.translationList[ aFullID ]
        bb = [0,0,0,0]

        aDict = self.__getSystemDict( aFullID )
        
        for fullID in aDict.keys():
            aType = getFullIDType( fullID )
            if aType == ME_SYSTEM_TYPE:
                bbchild = self.__createBB( fullID, aFullID )
            elif aType in [ ME_VARIABLE_TYPE, ME_PROCESS_TYPE ]:
                entityDotID = self.translationList[ fullID ]
                epos = self.attributeMap[entityDotID]["pos"]
                epos = map( lambda x: x*FACTOR, epos)
#                ewidth = self.attributeMap[entityDotID]["width"][0] *  POINTS_PER_INCH +40
                label = fullID.split(':')[2]
                ewidth, eheight = self.theShapePluginManager.getMinDims( aType, "Default", self.theGraphUtils, label )
                if aType == ME_VARIABLE_TYPE:
                    x = epos[0] - ewidth / 2 
                else:
                    x = epos[0] - eheight / 2
                y = epos[1] + 10
                x, y = self.__convertGVToAbsolute ( x, y )
                #relx, rely = self.__convertAbsToRel( absx, absy, theSystemObject )
                bbchild = [ x, y, x + ewidth, y + eheight ]
                self.attributeMap[entityDotID]["bb"] = bbchild[:]
            bb[0] = min( bb[0], bbchild[0] )
            bb[1] = min( bb[1], bbchild[1] )
            bb[2] = max( bb[2], bbchild[2] )
            bb[3] = max( bb[3], bbchild[3] )
        bb[0] -=20; bb[1] -= 30; bb[2] += 20; bb[3] += 20
        self.attributeMap[dotID]["bb"] = bb[:]
        return bb


    def __createConnections( self ):
        for aVarref in self.variableReferenceList:
            pobjectID = self.objectIDMap[ aVarref[0] ]
            vobjectID = self.objectIDMap[ aVarref[1] ]
            pdotID = self.translationList[aVarref[0]]
            vdotID = self.translationList[aVarref[1]]
            coef = float(aVarref[3])
            name = aVarref[2]
            '''
            if coef>0:
                edgeID = ":".join( [pdotID , vdotID, name ] )
            else:
                edgeID = ":".join( [vdotID , pdotID, name ]  )
            
            spline = self.edgeAttributeMap[ edgeID]["pos"]
            spline = map( lambda x:x*FACTOR, spline )
            coords=[]
            for i in range( 0, len(spline)/2 ):
                x, y = spline[i*2:i*2+2]
                absx, absy = self.__convertGVToAbsolute( x, y )
                coords.append( [absx,absy])
            inner = coords[2:]
            if coef > 0:
                #firs coord is var, order is ok
                orderedcoords = [ coords[1] ] + inner + [ coords[0] ]
                
            else:
                #first cooed is proce, order is reversed
                inner.reverse()
                orederedcoords = [ coords[0] ] + inner + [ coords[1] ]
            
            # get rings
            # get ringpositions variable
            # find least distant
            pobject = self.theLayout.getObject( pobjectID )
            vobject = self.theLayout.getObject( vobjectID )
            objectList = [pobject, vobject]
            epList = [ orderedcoords[0], orderedcoords[-1] ]
            ringList =[]
            for i in (0,1):
                minDist = [0,None]
                epx, epy = epList[i]
                for  ring in [ RING_TOP,RING_BOTTOM,RING_LEFT,RING_RIGHT ]:
                    x,y = objectList[i].getRingPosition(ring)
                    dist = math.sqrt( (epx-x)**2 + (epy-y)**2 )
                    if dist<minDist[0] or minDist[1] == None:
                        minDist[0] = dist
                        minDist[1] = ring
                ringList.append( minDist[1] )
            
            '''   
            (pring, vring) = self.theLayout.thePackingStrategy.autoConnect( pobjectID, vobjectID )
            anID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )

#            self.theLayout.createConnectionObject( anID, pobjectID, vobjectID,  ringList[0], ringList[1], None, aVarref[2] )
            self.theLayout.createConnectionObject( anID, pobjectID, vobjectID,  pring, vring, None, aVarref[2] )

            # substitue it with better interpolation!    
            '''
            firstControl = int(len(orderedcoords)/3)
            secondControl = int(len(orderedcoords)/3*2)
            orderedcoords = [ orderedcoords[0] , orderedcoords[firstControl] , orderedcoords[secondControl], orderedcoords[-1] ]

            ocM = nu.subtract( orderedcoords, orderedcoords[0] )
            try:
                ocM = list(nu.ravel(nu.divide( ocM, ocM[-1] )) )
            except:
                ocM = [0,0,0.3,0.3,0.6,0.6,1,1]
                '''
            aConnObject = self.theLayout.getObject( anID )
            
            aConnObject.setProperty( OB_SHAPE_TYPE, SHAPE_TYPE_CURVED_LINE )
#            aConnObject.setProperty( CO_CONTROL_POINTS, ocM )
            
        return
        
        
    def __convertGVToAbsolute( self, x, y ):
        return 0 + x, 0-y
        
    def __convertAbsToRel( self, absx, absy, system ):
        ix, iy = system.getAbsoluteInsidePosition()
        return absx-ix, absy-iy
        


