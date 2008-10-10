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

import numpy as n
import gtk
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.ComplexShape import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.LayoutCommand import *
from ecell.ui.model_editor.EntityCommand import *
from ecell.ui.model_editor.PackingStrategy import *
from ecell.ui.model_editor.LayoutBufferFactory import *

class EditorObject:
    def __init__(self, aLayout, objectID,  x, y, parentSystem ):
        self.theLayout = aLayout
        self.theID = objectID
        self.parentSystem = parentSystem

        self.thePropertyMap = {}
        self.thePropertyMap [ OB_SHAPE_TYPE ] = DEFAULT_SHAPE_NAME
        self.thePropertyMap[ OB_POS_X ] = x
        self.thePropertyMap[ OB_POS_Y ] = y
        self.theModelEditor=self.theLayout.theLayoutManager.theModelEditor
        self.ShapePluginManager=self.theModelEditor.getShapePluginManager()
        self.thePropertyMap[ OB_HASFULLID ] = False

        self.thePackingStrategy = PackingStrategy(self.theLayout)
        
        self.theCanvas = None
        self.theShape = None
        # default colors
        self.thePropertyMap [ OB_OUTLINE_COLOR ] = self.theLayout.graphUtils().getRRGByName("black")
        self.thePropertyMap [ OB_FILL_COLOR ] = self.theLayout.graphUtils().getRRGByName("white")
        self.thePropertyMap [ OB_TEXT_COLOR ] = self.theLayout.graphUtils().getRRGByName("blue")
        # default outline
        self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 1
        self.parentSystem.registerObject( self )
        #self.theSD = None
        self.isSelected = False

        self.existobjectFullIDList=[]
        
        self.thexorg=0
        self.theyorg=0
        self.lastx=0
        self.lasty=0
        self.rn=None
        self.totalLabelWidth=0
        self.labelLimit=0
        self.maxShiftMap={DIRECTION_LEFT:[0], DIRECTION_RIGHT:[2],DIRECTION_UP:[1],DIRECTION_DOWN:[3], 
                                  DIRECTION_BOTTOM_RIGHT:[2,3],DIRECTION_BOTTOM_LEFT:[0,3],DIRECTION_TOP_RIGHT:[2,1],   
                                  DIRECTION_TOP_LEFT:[0,1]}

    def getMinDims( self, theType, aLabel ):
        return self.ShapePluginManager.getMinDims( theType, "Default",self.getGraphUtils(), aLabel )
        
    def destroy(self):
        if self.theShape != None:
            self.theShape.delete()

    def hide( self ):
        # deletes it from canvas
        pass

    def doSelect( self, shift_pressed = False ):
        self.theLayout.selectRequest( self.theID, shift_pressed )

    def selected( self ):
        if not self.isSelected:
            self.isSelected = True
            self.theShape.selected()
            self.theShape.outlineColorChanged()
            
    
    
                

    def unselected( self ):
        if self.isSelected:
            self.isSelected = False
            self.theShape.unselected()
            self.theShape.outlineColorChanged()


    def outlineDragged( self, deltax, deltay, x, y ):
        if self.parentSystem.__class__.__name__ == 'Layout':
            return
        # in most of the cases object are not resizeable, only system is resizeable and it will override this
        self.objectDragged( )
        pass


    def popupEditor( self ):
        self.theLayout.popupObjectEditor( self.theID )      

    def setLimits( self, x0, y0, x1, y1 ):
        pass

    def setCanvas( self, aCanvas ):
        self.theCanvas = aCanvas

    def show( self ):
        self.theShape = ComplexShape( self, self.theCanvas, self.thePropertyMap[ OB_POS_X ], self.thePropertyMap[ OB_POS_Y ], self.thePropertyMap[ OB_DIMENSION_X ], self.thePropertyMap[ OB_DIMENSION_Y ] )
        
        self.theShape.show()

    def showLine(self):
        self.theLine = ComplexLine()
    
    
    def getShapeDescriptor(self, aShapeName ):
        graphUtils =  self.theLayout.graphUtils()
        theLabel = self.getProperty( OB_LABEL )
        aShapeType = self.getProperty( OB_TYPE )
        if aShapeName == aShapeType:
            aShapeName = "Default"
        return self.ShapePluginManager.createShapePlugin(aShapeType,aShapeName,self, graphUtils, theLabel)
        

    def getProperty( self, aPropertyName ):
        
        if aPropertyName in self.thePropertyMap.keys():
            return self.thePropertyMap[aPropertyName]
        else:
            raise Exception("Unknown property %s for object %s"%(aPropertyName, self.theID ) )


    def getPropertyList( self ):
        return self.thePropertyMap.keys()


    def getAbsolutePosition( self ):
        ( xpos, ypos ) = self.parentSystem.getAbsoluteInsidePosition()
        return ( xpos + self.thePropertyMap[ OB_POS_X ], ypos + self.thePropertyMap[ OB_POS_Y ]) 


    def getPropertyMap( self ):
        return self.thePropertyMap

    def setPropertyMap( self, aPropertyMap ):
        self.thePropertyMap = aPropertyMap
    

    def getID( self ):
        return self.theID

    def setProperty( self, aPropertyName, aPropertyValue):
        self.thePropertyMap[aPropertyName] = aPropertyValue
        if aPropertyName == OB_LABEL:
            self.theSD.renameLabel(aPropertyValue)
        elif aPropertyName in ( OB_DIMENSION_X, OB_DIMENSION_Y ):
            self.theSD.reCalculate()
        elif aPropertyName == OB_SHAPE_TYPE :
        
            anSD = self.getShapeDescriptor( aPropertyValue )
            self.setShapeDescriptor( anSD )

        if  self.theCanvas !=None:
            if aPropertyName == OB_LABEL:
                self.labelChanged(aPropertyValue)
            elif aPropertyName == OB_TEXT_COLOR :
                self.theShape.outlineColorChanged()
            elif aPropertyName == OB_OUTLINE_COLOR:
                self.theShape.outlineColorChanged()
            elif aPropertyName == OB_FILL_COLOR:
                self.theShape.fillColorChanged()
            
    def outOfRoot( self, x, y ):
        rootSystemID = self.theLayout.getProperty( LO_ROOT_SYSTEM )
        rootSystem = self.theLayout.getObject ( rootSystemID )
        rx1 = rootSystem.getProperty( OB_POS_X ) + 3
        ry1 = rootSystem.getProperty ( OB_POS_Y ) + 20
        rx2 = rx1 + rootSystem.getProperty( SY_INSIDE_DIMENSION_X )
        ry2 = ry1 + rootSystem.getProperty( SY_INSIDE_DIMENSION_Y )
        if x<rx1 or x>rx2 or y<ry1 or y>ry2:
            return True
        return False


    def setShapeDescriptor( self, anSD ):

        if self.theCanvas != None and self.theShape != None:
            self.theShape.delete()
        self.theSD = anSD
        self.thePropertyMap[OB_SHAPEDESCRIPTORLIST]=self.theSD
        
        if self.theCanvas != None:
            self.theShape.show()


    def labelChanged( self,aPropertyValue ):
        pass
        #newLabel = aPropertyValue.split(':')[2]
        #self.theShape.labelChanged(newLabel) 

    def getLayout( self ):
        return self.theLayout


    def getParent( self ):
        return self.parentSystem

    def getGraphUtils( self ):
        return self.theLayout.graphUtils()

    def getModelEditor( self ):
        return self.theLayout.theLayoutManager.theModelEditor

    def getCursorType( self, aFunction, x, y, buttonPressed ):
        if aFunction in [ SD_FILL, SD_TEXT ] and buttonPressed:
            return CU_MOVE
        elif aFunction  in [ SD_FILL, SD_TEXT ] and not buttonPressed:
            return CU_POINTER
        return CU_POINTER

    def move( self, deltax, deltay ):
        self.thePropertyMap[ OB_POS_X ] += deltax
        self.thePropertyMap[ OB_POS_Y ] += deltay
        if self.theShape != None:
            self.theShape.move( deltax, deltay )


    def parentMoved( self, deltax, deltay ):
        # no change in relative postions
        
        self.theShape.move( deltax, deltay )

    def adjustCanvas(self,dx,dy):
        if self.theCanvas.getBeyondCanvas():
            self.parentSystem.theCanvas.setCursor(CU_MOVE)
            self.parentSystem.theCanvas.scrollTo(dx,dy)
            self.parentSystem.theCanvas.setLastCursorPos(dx,dy)

    def getDirectionShift(self,dx,dy):
        if dx>0 and dy==0:
            return DIRECTION_RIGHT
        if dx<0 and dy==0:
            return DIRECTION_LEFT
        if dy>0 and dx==0:
            return DIRECTION_DOWN
        if dy<0 and dx==0:
            return DIRECTION_UP
        if dx>0 and dy>0:
            return DIRECTION_BOTTOM_RIGHT
        if dx<0 and dy>0:
            return DIRECTION_BOTTOM_LEFT
        if dx>0 and dy<0:
            return DIRECTION_TOP_RIGHT
        if dx<0 and dy<0:
            return DIRECTION_TOP_LEFT

    def adjustLayoutCanvas(self,dup,ddown,dleft,dright):
        scrollx=self.theLayout.getProperty(LO_SCROLL_REGION)[0]
        scrolly=self.theLayout.getProperty(LO_SCROLL_REGION)[1]
        scrollx2=self.theLayout.getProperty(LO_SCROLL_REGION)[2]
        scrolly2=self.theLayout.getProperty(LO_SCROLL_REGION)[3]
        dleft=-dleft
        dup=-dup
        scrollx+=dleft
        scrolly+=dup
        scrollx2+=dright
        scrolly2+=ddown
        self.theLayout.setProperty(LO_SCROLL_REGION,[scrollx,scrolly,scrollx2,scrolly2])
        self.theLayout.setProperty(OB_DIMENSION_X,scrollx2 - scrollx)
        self.theLayout.setProperty(OB_DIMENSION_Y,scrolly2 - scrolly)
        if self.theLayout.getCanvas() != None:
            self.theLayout.getCanvas().setSize(self.theLayout.getProperty(LO_SCROLL_REGION))
            self.theLayout.getCanvas().scrollTo(dleft+dright,ddown+dup) 

    ###################################################################################################
    def objectDragged( self, deltax, deltay ):
        cmdList=[]
        theParent=self.parentSystem
#        childs=len(theParent.getObjectList())
        #  parent system boundaries should be watched here!!!
        #get new positions:
        # currently move out of system is not supported!!!
        if theParent.__class__.__name__ == 'Layout':
            #rootsystem cannot be moved!!!
            return
        
        if self.theShape.getFirstDrag() and not self.theShape.getDragBefore() :
            self.rn=self.createRnOut()
            # store org position first
            self.thexorg= self.getProperty( OB_POS_X )
            self.theyorg= self.getProperty( OB_POS_Y )
            # for parent
            theParent.thex1org=theParent.thePropertyMap[OB_POS_X]
            theParent.they1org=theParent.thePropertyMap[OB_POS_Y]
            theParent.thex2org=theParent.thex1org+theParent.thePropertyMap[OB_DIMENSION_X]
            theParent.they2org=theParent.they1org+theParent.thePropertyMap[OB_DIMENSION_Y]
            # for siblings
#            if childs>1:
#                for sib in theParent.getObjectList():
#                    asib=theParent.getObject(sib)
#                    if asib.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):
#                        asib.thexorg=asib.getProperty(OB_POS_X)
#                        asib.theyorg=asib.getProperty(OB_POS_Y)
            #for Layout
            self.theLayout.orgScrollRegion = self.theLayout.getProperty( LO_SCROLL_REGION )
            self.theShape.setFirstDrag(False)
            self.theShape.setDragBefore(True)
            

        if self.theShape.getIsButtonPressed(): 
            newx = self.getProperty( OB_POS_X ) + deltax
            newy = self.getProperty( OB_POS_Y ) + deltay

            if newx==0 and newy == 0:
                return
            # get self's newx2,newy2
            newx2=newx+self.getProperty(OB_DIMENSION_X)
            newy2=newy+self.getProperty(OB_DIMENSION_Y)
            rpar=self.createRparent()
            if not self.isWithinParent(newx,newy,newx2,newy2,rpar):
                dir=self.getDirectionShift(deltax,deltay)
                UDLR=theParent.getUDLRmatrix(deltax,deltay,dir)
                deltaup=UDLR[theParent.UDLRMap['U']]
                deltadown=UDLR[theParent.UDLRMap['D']]
                deltaright=UDLR[theParent.UDLRMap['R']]
                deltaleft=UDLR[theParent.UDLRMap['L']]
                delta=theParent.getDelta(UDLR)
                
                if  theParent.parentSystem.__class__.__name__ == 'Layout':
                    self.adjustLayoutCanvas(   deltaup, deltadown, deltaleft, deltaright )
                    theParent.resize(   deltaup, deltadown, deltaleft, deltaright )
                else:
                    if dir in [DIRECTION_RIGHT,DIRECTION_UP,DIRECTION_DOWN,DIRECTION_LEFT]:
                        maxShift=theParent.getMaxShiftPos(dir)
                        if maxShift>0:
                            if maxShift>delta:              
                                theParent.resize(   deltaup, deltadown, deltaleft, deltaright )
                                self.adjustCanvas(deltax,deltay)
                            else:
                                return
                    else:
                        maxShiftx,maxShifty=theParent.getMaxShiftPos(dir)
                        if maxShiftx>0 and maxShifty>0:
                            if maxShiftx>delta and maxShifty>delta:         
                                theParent.resize(   deltaup, deltadown, deltaleft, deltaright )
                                self.adjustCanvas(deltax,deltay)
                            else:
                                return
                
            else: # stays within parent
                if self.isOverlap(newx,newy,newx2,newy2,self.rn):   
                    return                      
                else:           
                    self.move(deltax,deltay)
                    self.adjustCanvas(deltax,deltay)
        else: # button release
            #for self
            self.lastx=self.thePropertyMap[ OB_POS_X ]
            self.lasty=self.thePropertyMap[ OB_POS_Y ]
            newx=self.lastx
            newy=self.lasty
            self.move(-(self.getProperty( OB_POS_X )-self.thexorg) ,-(self.getProperty( OB_POS_Y )-self.theyorg))
            self.thePropertyMap[ OB_POS_X ] =self.thexorg
            self.thePropertyMap[ OB_POS_Y ] =self.theyorg

            #create command for self
            aCommand1 = MoveObject( self.theLayout, self.theID, newx, newy, None )
            #revCom = MoveObject( self.theLayout, self.theID, self.thexorg, self.theyorg )
            #aCommand1.makeExecuted([revCom])
            cmdList.append(aCommand1)

            #siblings
#            if childs>0:
#                for sib in theParent.getObjectList():
#                    asib=theParent.getObject(sib)
#                    if asib.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):
#                        asib.lastx=asib.thePropertyMap[ OB_POS_X ]
#                        asib.lasty=asib.thePropertyMap[ OB_POS_Y ]
#                        newxsib=asib.lastx
#                        newysib=asib.lasty
#                        asib.move(-(asib.getProperty( OB_POS_X )-asib.thexorg) ,-(asib.getProperty( OB_POS_Y )-asib.theyorg))
#                        asib.thePropertyMap[ OB_POS_X ] =asib.thexorg
#                        asib.thePropertyMap[ OB_POS_Y ] =asib.theyorg
#                        #create command for siblings
#                        aCommand= MoveObject( asib.theLayout, asib.theID, newxsib, newysib, None )
#                        cmdList.append(aCommand)


            #parent
            theParent.thedleftorg=theParent.thePropertyMap[OB_POS_X]-theParent.thex1org
            theParent.thedrightorg=theParent.thePropertyMap[OB_POS_X]+theParent.thePropertyMap[OB_DIMENSION_X]-theParent.thex2org
            theParent.theduporg=theParent.thePropertyMap[OB_POS_Y]-theParent.they1org
            theParent.theddownorg=theParent.thePropertyMap[OB_POS_Y]+theParent.thePropertyMap[OB_DIMENSION_Y]-theParent.they2org
            revCom2 = theParent.resize( theParent.theduporg, -theParent.theddownorg, theParent.thedleftorg, -theParent.thedrightorg )
            
            #create command for parent
            aCommand2 = ResizeObject( theParent.theLayout, theParent.theID, -theParent.theduporg, theParent.theddownorg, -theParent.thedleftorg, theParent.thedrightorg )
            #aCommand2.makeExecuted( [revCom2] )
            cmdList.append(aCommand2)
            

            # Layout
            newScrollRegion=self.theLayout.getProperty(LO_SCROLL_REGION)
            self.theLayout.setProperty(LO_SCROLL_REGION,self.theLayout.orgScrollRegion)
            #self.theLayout.setProperty(OB_DIMENSION_X,self.theLayout.orgScrollRegion[2]-self.theLayout.orgScrollRegion[0])
            #self.theLayout.setProperty(OB_DIMENSION_Y,self.theLayout.orgScrollRegion[3]-self.theLayout.orgScrollRegion[1])
            self.theLayout.getCanvas().setSize(self.theLayout.orgScrollRegion)
            #create command for Layout
            
            aCommand3 = ChangeLayoutProperty(self.theLayout, LO_SCROLL_REGION,newScrollRegion)
            revCom3 = ChangeLayoutProperty( self.theLayout, LO_SCROLL_REGION, self.theLayout.orgScrollRegion )
            #aCommand3.makeExecuted([revCom3])
            cmdList.append(aCommand3)
            self.theLayout.passCommand( cmdList)
            lastposx,lastposy=self.theLayout.getCanvas().getLastCursorPos()
            if self.theLayout.orgScrollRegion!=newScrollRegion:
                self.theLayout.getCanvas().scrollTo(lastposx,lastposy,'attach')
            

    def canPaste(self):
        return False
        

    def showMenu( self, anEvent, x=None,y=None ):
        self.newObjectPosX =x
        self.newObjectPosY = y
        menuDictList = self.getMenuItems()        
        aMenu = gtk.Menu()
        for i in range (len(menuDictList)):
            
            aMenuDict = menuDictList[i]
            if aMenuDict.keys() == []:
                return
            
            for aMenuName in aMenuDict.keys():
                menuItem = gtk.MenuItem( aMenuName )
                menuItem.connect( 'activate', aMenuDict[ aMenuName ] )
                
                if aMenuName =='undo':
                    
                    if not self.getModelEditor().canUndo() :
                        
                        menuItem.set_sensitive(False)
                if aMenuName =='redo':
                    
                    if not self.getModelEditor().canRedo():
                        menuItem.set_sensitive(False)

                if aMenuName =='paste':
                    if self.getModelEditor().getCopyBuffer() == None or not self.canPaste():
                        menuItem.set_sensitive(False)

                if aMenuName =='show system' or aMenuName =='show process' or aMenuName =='show variable':
                    (aSubMenu,NoOfSubMenuItem)=self.getSubMenu(aMenuName)
                    if NoOfSubMenuItem ==0:
                        menuItem.set_sensitive(False)
                    else:
                        menuItem.set_submenu(aSubMenu )
                if aMenuName == 'show connection':
                    (aSubMenu,NoOfSubMenuItem)=self.getConnectionMenu()
                    if NoOfSubMenuItem ==0:
                        menuItem.set_sensitive(False)
                    else:
                        menuItem.set_submenu(aSubMenu )

                if aMenuName == 'extend label':
                    maxShift=self.getMaxShiftPos(DIRECTION_RIGHT)
                    aLabel=self.getProperty(OB_LABEL)
                    if aLabel.endswith('...') and  maxShift>15: 
                        menuItem.set_sensitive(True)
                    else:
                        menuItem.set_sensitive(False)
                        

                aMenu.add(menuItem)
                

            aMenu.append( gtk.MenuItem() )
        aFullPNList = []
        for anObjectID in self.theLayout.theSelectedObjectIDList:
            anObject = self.theLayout.getObject( anObjectID )
            if anObject.getProperty( OB_HASFULLID):
                aFullPNList.append(anObject.getProperty( OB_FULLID ) +':')
        tracerSubMenu = self.getModelEditor().theRuntimeObject.createTracerSubmenu( aFullPNList )
        aMenu.append( tracerSubMenu )

        self.theMenu = aMenu
        aMenu.show_all()
        aMenu.popup(None, None, None, anEvent.button, anEvent.time)
            


    def getMenuItems( self, aSubMenu = None ):


        menuDict1 = {};menuDict2 = {};menuDict3 = {};menuDict4 = {}; menuDict5={}
        menuDict6 = {}

        menuDictList = []
        menuDict1['undo']=self.__undo
        menuDict1['redo']=self.__redo
        menuDict2['cut']=self.__cut
        menuDict2['copy']=self.__copy
        menuDict2['paste']=self.__paste
        menuDictList +=[menuDict1]
        menuDictList +=[menuDict2]
        if self.getProperty(OB_TYPE) == OB_TYPE_SYSTEM:
            menuDict5['show system'] = self.__test 
            menuDict5['show process'] = self.__test 
            menuDict5['show variable'] = self.__test 
            menuDictList +=[menuDict5]
        if self.parentSystem.__class__.__name__ != 'Layout' or self.getProperty(OB_TYPE) == OB_TYPE_CONNECTION:
            menuDict6 [ 'delete from layout'] = self.__userDeleteObject 
            if self.getProperty( OB_HASFULLID ) or self.getProperty(OB_TYPE) == OB_TYPE_CONNECTION:
                menuDict6['delete_from_model'] = self.__userDeleteEntity 
            menuDictList +=[menuDict6]
        if self.getProperty(OB_TYPE) == OB_TYPE_PROCESS or self.getProperty(OB_TYPE) == OB_TYPE_VARIABLE:
            menuDict4 [ 'show connection'] = self.__test
            menuDictList +=[menuDict4]
        if self.getProperty(OB_TYPE) != OB_TYPE_CONNECTION:
            menuDict3['extend label']=self.__extend_label
            menuDictList +=[menuDict3]

        return menuDictList
    
    

    def getSubMenuItems ( self, aMenuName):
        aModelEditor = self.getModelEditor()
        
        if self.parentSystem.__class__.__name__ == 'Layout':
            aParentFullID = 'System::/'
            aSystemPath='/'
        else:
            aParentFullID=self.getProperty(OB_FULLID)
            
            aSystemPath = convertSysIDToSysPath(aParentFullID)
        anEntityList=[]
        if aMenuName == 'show system':  
            aSystemEntityList = aModelEditor.theModelStore.getEntityList(ME_SYSTEM_TYPE,aSystemPath )
            for aSystem in aSystemEntityList:
                anEntityList+=[':'.join( [ME_SYSTEM_TYPE, aSystemPath, aSystem])]

        elif aMenuName == 'show process':
            aProcessEntityList=aModelEditor.theModelStore.getEntityList(ME_PROCESS_TYPE,aSystemPath )
            for aProcess in aProcessEntityList:
                anEntityList+=[':'.join( [ME_PROCESS_TYPE, aSystemPath, aProcess])] 
        
        elif aMenuName == 'show variable':
            aVariableEntityList=aModelEditor.theModelStore.getEntityList(ME_VARIABLE_TYPE,aSystemPath )
            for aVariable in aVariableEntityList:
                anEntityList+=[':'.join( [ME_VARIABLE_TYPE, aSystemPath, aVariable])] 

        
        return anEntityList

    def getConnectionMenu(self):
        aModelEditor = self.getModelEditor()
        aSubMenu = gtk.Menu() 
        NoOfSubMenuItem=0

        #check whether the var or pro is in the layout
        existObjectList =self.theLayout.getObjectList()
    
        #get the object FullID exist in the layout using its objectID
        existObjectFullIDList = []
        for anID in existObjectList:
            object = self.theLayout.getObject(anID)
            if object.getProperty(OB_HASFULLID):
                objectFullID = object.getProperty(OB_FULLID)
                existObjectFullIDList += [[objectFullID,anID]]

        if self.getProperty(OB_TYPE)==OB_TYPE_PROCESS:
            
            #get process obj varrReff list from modelstore by passing the FullID
            aProcessFullID = self.getProperty( OB_FULLID )
            aVarReffList = aModelEditor.theModelStore.getEntityProperty(aProcessFullID+':' +MS_PROCESS_VARREFLIST)
            #convert the relative path of var full id into the absolute
            aVarReffList1 =[]       
            for aVarReff in aVarReffList:
                varFullID = getAbsoluteReference(aProcessFullID, aVarReff[ME_VARREF_FULLID])
                aVarReffList1 +=[[aVarReff[ME_VARREF_NAME], varFullID]]

            #get list of connection return the connId
            connectionList = self.getProperty(PR_CONNECTIONLIST)
        
            #get the connectionobj for each conn id
            aVarReffList2 =[]
            for conn in connectionList:
                connObj = self.theLayout.getObject( conn )
                #get var attached to n varref name for each conn
                varreffName = connObj.getProperty(CO_NAME)
                varID = connObj.getProperty(CO_VARIABLE_ATTACHED)
                #get var FUllID
                if varID!=None:
                    varFullID = self.theLayout.getObject( varID ).getProperty(OB_FULLID)
                aVarReffList2+=[varreffName]
            
            #check if there is missing variable
            ExistVarRefNameList = []
            if len(aVarReffList1)!=len(aVarReffList2) :
                for i in range (len(aVarReffList1)):
                    aVar = aVarReffList1[i][ME_VARREF_FULLID] 
                    aVarReff = aVarReffList1[i][ME_VARREF_NAME] 
                    if not aVarReff in aVarReffList2:
                        for j in range (len(existObjectFullIDList)): 
                            if aVar ==existObjectFullIDList[j][0] and  aVarReff not in ExistVarRefNameList :
                                menuItem = gtk.MenuItem( aVar+': '+aVarReff )
                                menuItem.set_name( aVar+','+aVarReff + ',' +existObjectFullIDList[j][1] )
                                ExistVarRefNameList +=[aVarReff]
                                menuItem.connect( 'activate', self.__userCreateConnection )
                                NoOfSubMenuItem+=1
                                aSubMenu.add(menuItem)
            

        if self.getProperty(OB_TYPE)==OB_TYPE_VARIABLE:
            #get var obj process list from modelstore by passing the FullID
            aVariableFullID = self.getProperty( OB_FULLID )
            aProcessList = aModelEditor.theModelStore.getEntityProperty(aVariableFullID+':' +MS_VARIABLE_PROCESSLIST)
            #get list of connection return the connId
            connectionList = self.getProperty(VR_CONNECTIONLIST)
            #get the connectionobj for each conn id
            aProcessList2 =[]
            for conn in connectionList:
                connObj = self.theLayout.getObject( conn )
                varreffName = connObj.getProperty(CO_NAME)
                proID = connObj.getProperty(CO_PROCESS_ATTACHED)
                proFullID = self.theLayout.getObject( proID ).getProperty(OB_FULLID)
                aProcessList2+=[proFullID]

            #check if there is missing process
            existProList =[]
            if len(aProcessList)!=len(aProcessList2) :
                for aPro in aProcessList :
                    if not aPro in aProcessList2:
                        for j in range (len(existObjectFullIDList)): 
                            if aPro ==existObjectFullIDList[j][0] and not aPro in existProList:
                                menuItem = gtk.MenuItem( aPro )
                                menuItem.set_name( aPro+','+existObjectFullIDList[j][1] )
                                existProList += [aPro]
                                menuItem.connect( 'activate', self.__userCreateConnection )
                                NoOfSubMenuItem+=1
                                aSubMenu.add(menuItem)
            
        return (aSubMenu,NoOfSubMenuItem)
                    
        
    
    def getSubMenu(self, aMenuName ):

        anEntityList = self.getSubMenuItems(aMenuName)

        #set the self.existobjectFullIDList
        self.setExistObjectFullIDList()
        aSubMenu = gtk.Menu() 
        NoOfSubMenuItem=0
        for aSubMenuName in anEntityList:
            if not aSubMenuName in self.existobjectFullIDList:
                menuItem = gtk.MenuItem( aSubMenuName )
                menuItem.set_name( aSubMenuName)
                menuItem.connect( 'activate', self.__userCreateObject )
                NoOfSubMenuItem+=1
                aSubMenu.add(menuItem)
                
        self.existobjectFullIDList = [] 
            
        return (aSubMenu,NoOfSubMenuItem)

        
    def ringDragged( self, shapeName, deltax, deltay ):
        pass

        
    def pasteObject(self):
        pass
    
    def getParent( self ):
        return self.parentSystem

    def getModelEditor( self ):
        return self.theLayout.theLayoutManager.theModelEditor


    def __test(self, *args ):
        pass

    def __userDeleteObject( self, *args ):
        aCommandList = []
        self.theMenu.destroy()
        self.theLayout.deleteSelected()

        
        
    def __userDeleteEntity ( self, *args ):

        self.theMenu.destroy()
        self.theLayout.deleteEntities()
            
            

    def __undo(self, *args ):
        self.getModelEditor().undoCommandList()
        

    def __redo(self, *args ):
        self.getModelEditor().redoCommandList()
        GraphUtils
    def __cut(self,*args):
        
        self.__copy( None )
        self.__userDeleteEntity( None )
        
    def __copy(self,*args):
        
        if self.parentSystem.__class__.__name__ != 'Layout':
            aLayoutManager = self.getModelEditor().theLayoutManager
            self.LayoutBufferFactory =  LayoutBufferFactory(self.getModelEditor(), aLayoutManager)
            self.aLayoutBuffer = self.LayoutBufferFactory.createMultiObjectBuffer(self.theLayout.theName, \
                    self.theLayout.theSelectedObjectIDList )
            self.getModelEditor().setCopyBuffer(self.aLayoutBuffer )


    def __paste(self,*args):
        self.pasteObject()


    def __userCreateConnection(self,*args):
        if self.getProperty(OB_TYPE) == OB_TYPE_PROCESS:
            if len(args) == 0:
                return None
            if type( args[0] ) == gtk.MenuItem:
                variableID = args[0].get_name()
            varrefName = variableID.split(',')[1]
            variableID = variableID.split(',')[2]
            newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
            (processRing, variableRing) = self.thePackingStrategy.autoConnect(self.theID, variableID )
            aCommand = CreateConnection( self.theLayout, newID,  self.theID,variableID, processRing, variableRing, PROCESS_TO_VARIABLE, varrefName )
            
            self.theLayout.passCommand([aCommand])
        
            

        if self.getProperty(OB_TYPE)==OB_TYPE_VARIABLE:
            aVariableFullID = self.getProperty(OB_FULLID)
            if len(args) == 0:
                return None
            if type( args[0] ) == gtk.MenuItem:
                processID = args[0].get_name()
            aProcessFullID = processID.split(',')[0]
            processID = processID.split(',')[1]

            #get var reff name
            aModelEditor = self.getModelEditor()
            aVarReffList = aModelEditor.theModelStore.getEntityProperty(aProcessFullID+':' +MS_PROCESS_VARREFLIST)
            varReffNameList = []
            for i in range (len(aVarReffList)):
                varFullID= getAbsoluteReference(aProcessFullID, aVarReffList[i][ME_VARREF_FULLID])
                if aVariableFullID ==varFullID:
                    varReffNameList+=[aVarReffList[i][ME_VARREF_NAME]]
            for avarRefName in varReffNameList:
                newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
                (processRing, variableRing) = self.thePackingStrategy.autoConnect( processID, self.theID )
                aCommand = CreateConnection( self.theLayout, newID,  processID,self.theID, processRing, variableRing, PROCESS_TO_VARIABLE, avarRefName )
                self.theLayout.passCommand(  [aCommand]  )  
               
        

        

    def __userCreateObject(self,*args):
        
        (offsetx, offsety ) = self.getAbsolutePosition()
        x = self.newObjectPosX - (self.theSD.insideX + offsetx )
        y = self.newObjectPosY - ( self.theSD.insideY + offsety )

        
        aModelEditor = self.theLayout.theLayoutManager.theModelEditor
        
        
        
        if len(args) == 0:
            return None
        if type( args[0] ) == gtk.MenuItem:
            anEntityName = args[0].get_name()

        aFullID = anEntityName
        objectType = aFullID.split(':')[0]
        objectID = self.theLayout.getUniqueObjectID( objectType)
        
         
        #create aCommand
        aCommand = None
    
        if not aFullID in self.existobjectFullIDList:
            aCommand = CreateObject( self.theLayout, objectID, objectType, aFullID, x,y, self )         
                

        if aCommand != None:
            px2=self.getProperty(OB_DIMENSION_X)
            py2=self.getProperty(OB_DIMENSION_Y)
            rpar=n.array([0,0,px2,py2])
            if objectType == OB_TYPE_SYSTEM:
                rn=self.createRnAddSystem()
                minreqx, minreqy = self.getMinDims( ME_SYSTEM_TYPE, aFullID )
                x2=x+max( SYS_MINWIDTH, minreqx )
                y2=y+max( SYS_MINHEIGHT, minreqy )
                availspace=self.getAvailSpace(x,y,x2,y2,rn)
                self.availSpace=availspace
                # check boundaries
                if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                    self.theLayout.passCommand( [aCommand] )
                    
            if objectType == OB_TYPE_PROCESS:
                minreqx, minreqy = self.getMinDims( ME_PROCESS_TYPE, aFullID.split(":")[2] )
                x2=x+max( PRO_MINWIDTH, minreqx )
                y2=y+max( PRO_MINHEIGHT, minreqy )
                rn=self.createRnAddOthers()
                if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                    self.theLayout.passCommand( [aCommand] )

            if objectType == OB_TYPE_VARIABLE:
                minreqx, minreqy = self.getMinDims( ME_VARIABLE_TYPE, aFullID.split(":")[2] )
                x2=x+max( VAR_MINWIDTH, minreqx )
                y2=y+max( VAR_MINHEIGHT, minreqy )
                # check boundaries
                rn=self.createRnAddOthers()
                if (not self.isOverlap(x,y,x2,y2,rn) and self.isWithinParent(x,y,x2,y2,rpar)):
                    self.theLayout.passCommand( [aCommand] )
                    
            else:
                pass


    def __extend_label(self,*args):
        aLabel=self.getProperty(OB_LABEL)
        oldLen=len(aLabel)
        aFullID=self.getProperty(OB_FULLID)
        idLen=len(aFullID)
        aType=self.getProperty(OB_TYPE)
        if aType!=OB_TYPE_SYSTEM :
            aFullID=aFullID.split(':')[2]
        maxShift=self.getMaxShiftPos(DIRECTION_RIGHT)
        newLabel=aFullID[0:oldLen]
        self.calcLabelParam(newLabel)
        totalWidth,limit=self.getLabelParam()
        while totalWidth<limit and oldLen<=len(aFullID):
            oldLen+=1
            newLabel=aFullID[0:oldLen]
            self.calcLabelParam(newLabel)
            totalWidth,limit=self.getLabelParam()
        if newLabel!=aFullID:
            newLabel=newLabel[0:len(newLabel)-3]+'...'
        newDimx=self.estLabelWidth(newLabel)
        oldDimx=self.getProperty(OB_DIMENSION_X)
        deltaWidth=newDimx-oldDimx
        resizeCommand = ResizeObject(self.getLayout(),self.getID(), 0, 0, 0, deltaWidth)            
        relabelCommand = SetObjectProperty( self.getLayout(), self.getID(), OB_LABEL, newLabel )
        self.getLayout().passCommand( [resizeCommand,relabelCommand] )  


    
    def setExistObjectFullIDList(self):
        #get the existing objectID in the System
        exsistObjectInTheLayoutList = []
        if self.getProperty(OB_HASFULLID):
            exsistObjectInTheLayoutList =self.getObjectList()

        #get the object FullID exist in the layout using its objectID
        for anID in exsistObjectInTheLayoutList:
            object = self.theLayout.getObject(anID)
            objectFullID = object.getProperty(OB_FULLID)
            self.existobjectFullIDList += [objectFullID]
    

    def createRnOut(self):
        no=len(self.parentSystem.getObjectList())
        rn=None
        if no>1:
            for sib in self.parentSystem.getObjectList():
                asib=self.parentSystem.getObject(sib)
                if (asib.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID)) and (asib.getProperty(OB_TYPE)==OB_TYPE_SYSTEM):
                    asibx1=asib.getProperty(OB_POS_X)
                    asiby1=asib.getProperty(OB_POS_Y)
                    asibx2=asibx1+asib.getProperty(OB_DIMENSION_X)
                    asiby2=asiby1+asib.getProperty(OB_DIMENSION_Y)
                    rsib=n.array([asibx1,asiby1,asibx2,asiby2])
                    rsib=n.reshape(rsib,(4,1))
                    if rn==None:
                        rn=rsib
                    else:
                        rn=n.concatenate((rn,rsib),1)
        return rn

    

    
    def isOverlap(self,x1,y1,x2,y2,rn):
        r1 = n.array([x1,y1,x2,y2])
        r1=n.reshape(r1,(4,1))
        if rn!=None:
            return self.getGraphUtils().calcOverlap(r1,rn)
        else:
            return False
    
        
    def isWithinParent(self,u1,v1,u2,v2,rpar):
        rpar=n.reshape(rpar,(4,1))
        olw = self.getProperty( OB_OUTLINE_WIDTH )
        #v2+=olw*8 #height of the parent label
        r2 = n.array([u1,v1,u2,v2])
        r2=n.reshape(r2,(4,1))
        return self.getGraphUtils().calcWithin(rpar,r2)

    def getGraphUtils( self ):
        return self.theLayout.graphUtils()


    def buttonReleased( self ):
        pass

    def createRparent(self):
        olw=self.thePropertyMap[ OB_OUTLINE_WIDTH ]
        if self.parentSystem.__class__.__name__ == 'Layout':
            x1=0
            y1=0
            x2= x1+self.parentSystem.getProperty(OB_DIMENSION_X)
            y2= y1+self.parentSystem.getProperty(OB_DIMENSION_Y)
        else:
            x1=0
            #y1=self.parentSystem.getProperty(OB_DIMENSION_Y)-self.parentSystem.getProperty
#(SY_INSIDE_DIMENSION_Y)
            y1=0
            x2= x1+self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)
            y2= y1+self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)
        r1=n.array([x1,y1,x2,y2])
        return r1  
    
    def estLabelWidth(self,newLabel):
        pass

    def getAvailableShapes( self ):
        return self.ShapePluginManager.getShapeList(self.getProperty( OB_TYPE ) )
    

    def setLabelParam(self,totalWidth,limit):
        self.totalLabelWidth=totalWidth
        self.totalLimit=limit

    def getLabelParam(self):
        return self.totalLabelWidth,self.totalLimit

    def calcLabelParam(self,label):
        estWidth=self.estLabelWidth(label)
        newx2=estWidth
        #check overlap and within parent
        x=self.getProperty(OB_POS_X)
        x2=self.getProperty(OB_DIMENSION_X)
        totalWidth=x+newx2
        maxShift=self.getMaxShiftPos(DIRECTION_RIGHT)
        limit=x+x2+maxShift
        self.setLabelParam(totalWidth,limit)
        return maxShift + x2


    def truncateLabel(self,aLabel,lblWidth,dimx):
        truncatedLabel=self.getGraphUtils().truncateLabel(aLabel,lblWidth,dimx,self.getProperty(OB_MINLABEL))
        return truncatedLabel

    def getMaxShiftPos(self,direction):
        dir=direction
        olw=self.getProperty(OB_OUTLINE_WIDTH)
        x1=self.getProperty(OB_POS_X)
        y1=self.getProperty(OB_POS_Y)
        x2=x1+self.getProperty(OB_DIMENSION_X)
        y2=y1+self.getProperty(OB_DIMENSION_Y)
        r1=n.array([x1,y1,x2,y2])
        rn=self.createRnOut()
        rpar=self.createRparent()
        matrix = self.getGraphUtils().calcMaxShiftPos(r1,rn,dir,rpar)
        mshift=matrix-r1
        mshift = mshift * n.array( [-1,-1,1,1] )
        if len(self.maxShiftMap[dir])>1:
            posx,posy=self.maxShiftMap[dir][0],self.maxShiftMap[dir][1]
            return max(0, mshift[posx] ), max(0, mshift[posy] )
        else:
            pos=self.maxShiftMap[dir][0]
            return max(0, mshift[pos] )

class GhostLine:
    def __init__( self, parentObject, ringCode, endx, endy ):
        self.parentObject = parentObject
        self.x2 = endx
        self.y2 = endy
        (self.x1, self.y1) = self.parentObject.getRingPosition( ringCode )
        self.root = self.parentObject.theCanvas.getCanvas().root()
        self.theLine = self.root.add( gnomecanvas.CanvasLine, points=(self.x1, self.y1, self.x2, self.y2), last_arrowhead = True,arrow_shape_a=5, arrow_shape_b=5, arrow_shape_c=3 )

    def moveEndPoint( self, deltax, deltay ):
        if self.parentObject.outOfRoot( self.x2 + deltax, self.y2 + deltay ):
            return
        self.x2 += deltax
        self.y2 += deltay
        self.theLine.set_property( "points", (self.x1, self.y1, self.x2, self.y2) )

    def getEndPoint( self ):
        return ( self.x2, self.y2 )

    def registerObject( self, anObject ):
        self.theObjectMap[anObject.getID()] = anObject

    def unregisterObject ( self, anObjectID ):
        self.theObjectMap.__delitem__( anObjectID )

    def delete ( self ):
        self.theLine.destroy()
