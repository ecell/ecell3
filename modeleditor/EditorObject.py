
from ComplexShape import *
from Constants import *
from LayoutCommand import *
import gtk
from EntityCommand import *
from Utils import *
from PackingStrategy import *



class EditorObject:


	def __init__(self, aLayout, objectID,  x, y, parentSystem ):
		self.theLayout = aLayout
		self.theID = objectID
		self.parentSystem = parentSystem
		
		self.thePropertyMap = {}
		self.thePropertyMap[ OB_POS_X ] = x
		self.thePropertyMap[ OB_POS_Y ] = y
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

		
	def destroy(self):
		if self.theShape != None:
			self.theShape.delete()

	def hide( self ):
		# deletes it from canvas
		pass

	def doSelect( self ):
		if not self.isSelected:
			self.theLayout.selectRequest( self.theID )

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
		
		if  self.theCanvas !=None:
			if aPropertyName == OB_FULLID:
				self.labelChanged(aPropertyValue)
			elif aPropertyName == OB_TEXT_COLOR :
				self.theShape.outlineColorChanged()
			elif aPropertyName == OB_OUTLINE_COLOR:
				self.theShape.outlineColorChanged()
			elif aPropertyName == OB_FILL_COLOR:
				self.theShape.fillColorChanged()

	def labelChanged( self,aPropertyValue ):
		newLabel = aPropertyValue.split(':')[2]
		self.theShape.labelChanged(newLabel) 
		

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
		self.theShape.move( deltax, deltay )

	def parentMoved( self, deltax, deltay ):
		# no change in relative postions

		self.theShape.move( deltax, deltay )

	def adjustCanvas(self,dx,dy):
		if self.theCanvas.getBeyondCanvas():
			self.parentSystem.theCanvas.setCursor(CU_MOVE)
			self.parentSystem.theCanvas.scrollTo(dx,dy)
	
	###################################################################################################
	def objectDragged( self, deltax, deltay ):
		theParent=self.parentSystem
		#  parent system boundaries should be watched here!!!
		#get new positions:
		# currently move out of system is not supported!!!
		if theParent.__class__.__name__ == 'Layout':
			#rootsystem cannot be moved!!!
			return
		
		if self.theShape.getFirstDrag() and not self.theShape.getDragBefore() :
			# store org position first
			self.thexorg= self.getProperty( OB_POS_X )
			self.theyorg= self.getProperty( OB_POS_Y )
			self.theShape.setFirstDrag(False)
			self.theShape.setDragBefore(True)

		if self.theShape.getIsButtonPressed(): 
			newx = self.getProperty( OB_POS_X ) + deltax
			newy = self.getProperty( OB_POS_Y ) + deltay

		elif not self.theShape.getIsButtonPressed():
			self.lastx=self.thePropertyMap[ OB_POS_X ]
			self.lasty=self.thePropertyMap[ OB_POS_Y ]
			newx=self.lastx
			newy=self.lasty
			self.move(-(self.getProperty( OB_POS_X )-self.thexorg) ,-(self.getProperty( OB_POS_Y )-self.theyorg))
			self.thePropertyMap[ OB_POS_X ] =self.thexorg
			self.thePropertyMap[ OB_POS_Y ] =self.theyorg
			
		if newx==0 and newy == 0:
			return
		
		# get self's newx2,newy2
		newx2=newx+self.getProperty(OB_DIMENSION_X)
		newy2=newy+self.getProperty(OB_DIMENSION_Y)
		
		if theParent.isWithinParent(newx,newy,self.getProperty(OB_TYPE),self):
			for aKey in theParent.getObjectList():
				aParChild=self.parentSystem.theLayout.getObject(aKey)
				if aParChild.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):	
					childX1=aParChild.getProperty(OB_POS_X)
					childX2=childX1+aParChild.getProperty(OB_DIMENSION_X)
					childY1=aParChild.getProperty(OB_POS_Y)
					childY2=childY1+aParChild.getProperty(OB_DIMENSION_Y)
						
					# check for overlapping
					if self.getProperty(OB_TYPE)==OB_TYPE_SYSTEM:
						if theParent.isOverlap(newx,newy,newx2,newy2,childX1,childY1,childX2,childY2):	
							return
					else:
						if theParent.isOverlapSystem(newx,newy,self.getProperty(OB_TYPE),self):
							return	
			if self.theShape.getIsButtonPressed():
				self.move(deltax,deltay)
			elif not self.theShape.getIsButtonPressed():
				aCommand = MoveObject( self.theLayout, self.theID, newx, newy, None )
				self.theLayout.passCommand( [ aCommand ] )
			self.adjustCanvas(deltax,deltay)
		else:
			return
		
	

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
						menuItem.set_sensitive(gtk.FALSE)
				if aMenuName =='redo':
					
					if not self.getModelEditor().canRedo():
						menuItem.set_sensitive(gtk.FALSE)
					
				if aMenuName =='show system' or aMenuName =='show process' or aMenuName =='show variable':
					(aSubMenu,NoOfSubMenuItem)=self.getSubMenu(aMenuName)
					if NoOfSubMenuItem ==0:
						menuItem.set_sensitive(gtk.FALSE)
					else:
						menuItem.set_submenu(aSubMenu )
				if aMenuName == 'show connection':
					(aSubMenu,NoOfSubMenuItem)=self.getConnectionMenu()
					if NoOfSubMenuItem ==0:
						menuItem.set_sensitive(gtk.FALSE)
					else:
						menuItem.set_submenu(aSubMenu )

				aMenu.add(menuItem)
				

			aMenu.append( gtk.MenuItem() )
			
		
		self.theMenu = aMenu
		aMenu.show_all()
		aMenu.popup(None, None, None, anEvent.button, anEvent.time)
			

	def getMenuItems( self, aSubMenu = None ):
		menuDict1 = {};menuDict2 = {};menuDict3 = {};menuDict4 = {}
		menuDictList = []
		menuDict1['undo']=self.__undo
		menuDict1['redo']=self.__redo
		menuDictList +=[menuDict1]
		if self.getProperty(OB_TYPE) == OB_TYPE_SYSTEM:
			menuDict2['show system'] = self.__test 
			menuDict2['show process'] = self.__test 
			menuDict2['show variable'] = self.__test 
			menuDictList +=[menuDict2]
		if self.parentSystem.__class__.__name__ != 'Layout' or self.getProperty(OB_TYPE) == OB_TYPE_CONNECTION:
			menuDict3 [ 'delete from layout'] = self.__userDeleteObject 
			if self.getProperty( OB_HASFULLID ) or self.getProperty(OB_TYPE) == OB_TYPE_CONNECTION:
				menuDict3['delete_from_model'] = self.__userDeleteEntity 
			menuDictList +=[menuDict3]
		if self.getProperty(OB_TYPE) == OB_TYPE_PROCESS or self.getProperty(OB_TYPE) == OB_TYPE_VARIABLE:
			menuDict4 [ 'show connection'] = self.__test
			menuDictList +=[menuDict4]
	
		
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
				varFullID = getAbsoluteReference(aProcessFullID, aVarReff[1])
				aVarReffList1 +=[[varFullID,aVarReff[0]]]

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
			if len(aVarReffList1)!=len(aVarReffList2) :
				for i in range (len(aVarReffList1)):
					aVar = aVarReffList1[i][0] 
					aVarReff = aVarReffList1[i][1] 
					if not aVarReff in aVarReffList2:
						for j in range (len(existObjectFullIDList)): 
							if aVar ==existObjectFullIDList[j][0]:
								menuItem = gtk.MenuItem( aVar+': '+aVarReff )
								menuItem.set_name( aVar+','+aVarReffList1[i][1] + ',' +existObjectFullIDList[j][1] )
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
			if len(aProcessList)!=len(aProcessList2) :
				for aPro in aProcessList :
					if not aPro in aProcessList2:
						for j in range (len(existObjectFullIDList)): 
							if aPro ==existObjectFullIDList[j][0]:
								menuItem = gtk.MenuItem( aPro )
								menuItem.set_name( aPro+','+existObjectFullIDList[j][1] )
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
		
	
	def getParent( self ):
		return self.parentSystem

	def getModelEditor( self ):
		return self.theLayout.theLayoutManager.theModelEditor





	def __test(self, *args ):
		
		pass

	def __userDeleteObject( self, *args ):
		self.theMenu.destroy()
		
		
#		if self.getProperty( OB_TYPE )==OB_TYPE_PROCESS : 
#			connectionList = self.getProperty(PR_CONNECTIONLIST)
#			for conn in connectionList:
#				aCommand= DeleteObject( self.theLayout,conn )
#				self.theLayout.passCommand( [ aCommand ] )
#
#		if self.getProperty( OB_TYPE )==OB_TYPE_VARIABLE:
#			connectionList = self.getProperty(VR_CONNECTIONLIST)
#			for conn in connectionList:
#				aCommand= DeleteObject( self.theLayout,conn )
#				self.theLayout.passCommand( [ aCommand ] )
				
		aCommand = DeleteObject( self.theLayout, self.theID )
		self.theLayout.passCommand( [ aCommand ] )
		
		
	def __userDeleteEntity ( self, *args ):

		self.theMenu.destroy()
		aModelEditor = self.theLayout.theLayoutManager.theModelEditor
		if self.getProperty( OB_HASFULLID ):
			aFullID = self.getProperty( OB_FULLID )
			aCommand = DeleteEntityList( aModelEditor, [aFullID ] )
			self.theLayout.passCommand( [ aCommand ] )

		elif self.getProperty( OB_TYPE )==OB_TYPE_CONNECTION:
			connObj = self.theLayout.getObject( self.theID )
			varreffName = connObj.getProperty(CO_NAME)
			proID = connObj.getProperty(CO_PROCESS_ATTACHED)
			aProcessObject = self.theLayout.getObject(proID)
			aProcessFullID = aProcessObject.getProperty( OB_FULLID )
			fullPN = aProcessFullID+':' +MS_PROCESS_VARREFLIST
			aVarReffList = copyValue( aModelEditor.theModelStore.getEntityProperty( fullPN ) )
			for i in range (len(aVarReffList)): 
				if varreffName == aVarReffList[i][0]:
					del aVarReffList[i]
					break
			aCommand = ChangeEntityProperty( aModelEditor, fullPN, aVarReffList )

			self.theLayout.passCommand( [ aCommand ] )
		

	def __undo(self, *args ):
		self.getModelEditor().undoCommandList()
		

	def __redo(self, *args ):
		self.getModelEditor().redoCommandList()
		

	def __userCreateConnection(self,*args):
		if self.getProperty(OB_TYPE)==OB_TYPE_PROCESS:
			if len(args) == 0:
				return None
			if type( args[0] ) == gtk.MenuItem:
				variableID = args[0].get_name()
			varrefName = variableID.split(',')[1]
			variableID = variableID.split(',')[2]
		
			aCommandList = self.thePackingStrategy.autoConnect( self.theID, variableID,varrefName )
			self.theLayout.passCommand(  aCommandList  )
		
			

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
				varFullID= getAbsoluteReference(aProcessFullID, aVarReffList[i][1])
				if aVariableFullID ==varFullID:
					varReffNameList+=[aVarReffList[i][0]]
			for avarRefName in varReffNameList:
				aCommandList = self.thePackingStrategy.autoConnect( processID, self.theID, avarRefName )
				self.theLayout.passCommand(  aCommandList  )	
		       
		

		

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
			aCommand=CreateObject( self.theLayout, objectID, objectType, aFullID, x,y, self )         
                

		if aCommand != None:

			if objectType == OB_TYPE_SYSTEM:
				# check boundaries
				maxAvailWidth,maxAvailHeight=self.getMaxDimensions(x,y)
				if ((maxAvailWidth>=SYS_MINWIDTH and maxAvailHeight>=SYS_MINHEIGHT) and self.isWithinParent(x,y,OB_TYPE_SYSTEM,None)):
					self.theLayout.passCommand( [aCommand] )
					
			if objectType == OB_TYPE_PROCESS:
				# check boundaries
				if (not self.isOverlapSystem(x,y,OB_TYPE_PROCESS,None) and self.isWithinParent(x,y,OB_TYPE_PROCESS,None)):
					self.theLayout.passCommand( [aCommand] )
					

			if objectType == OB_TYPE_VARIABLE:
				# check boundaries
				if (not self.isOverlapSystem(x,y,OB_TYPE_VARIABLE,None) and self.isWithinParent(x,y,OB_TYPE_VARIABLE,None)):
					self.theLayout.passCommand( [aCommand] )
					
		
			else:
				pass


	
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
	
	

	###########################################################################################
	###########################################################################################

	#######################################
	#  a system is being resized inwardly #
	#######################################
	def getMaxShiftNeg(self,direction):
		olw=self.getProperty(OB_OUTLINE_WIDTH)
		maxAvailShiftNeg=0
		x1List=[]
		y1List=[]
		x2List=[]
		y2List=[]
		if self.getProperty(OB_TYPE)=='System':
			sysX1=0
			sysY1=0
			sysX2=sysX1+self.getProperty(OB_DIMENSION_X)
			sysY2=sysY1+self.getProperty(OB_DIMENSION_Y)
			textWidth=self.getProperty(OB_SHAPEDESCRIPTORLIST).getRequiredWidth()		
			# check if it has any child
			if len(self.getObjectList())>0:
				for aKey in self.getObjectList():
					aChildObj=self.theLayout.getObject(aKey)
					childX1=aChildObj.getProperty(OB_POS_X)
					childX2=childX1+aChildObj.getProperty(OB_DIMENSION_X)
					childY1=aChildObj.getProperty(OB_POS_Y)
					childY2=childY1+aChildObj.getProperty(OB_DIMENSION_Y)+olw*8
					x1List.append(childX1)
					y1List.append(childY1)
					x2List.append(childX2)
					y2List.append(childY2)
					x1List.sort()
					y1List.sort()
					x2List.sort()
					y2List.sort()
				# bottom or top
				if direction == DIRECTION_DOWN or direction==DIRECTION_UP:
					maxAvailShiftNeg=(sysY2-(y2List[len(y2List)-1]))
				# right or left
				elif direction == DIRECTION_RIGHT or direction==DIRECTION_LEFT:
					# check against ID
					if textWidth>x2List[len(x2List)-1]:
						maxAvailShiftNeg=(sysX2-textWidth)		
					else:
						maxAvailShiftNeg=(sysX2-x2List[len(x2List)-1])
			else:
				# no child
				# bottom or top
				if direction == DIRECTION_DOWN or direction==DIRECTION_UP:
					maxAvailShiftNeg=(sysY2-sysY1)-olw*10
				# right or left
				elif direction == DIRECTION_RIGHT or direction==DIRECTION_LEFT:
					maxAvailShiftNeg=sysX2-textWidth
		return maxAvailShiftNeg


	
	#########################################
	#  a system is being resized outwardly  #
	#########################################
	def getMaxShiftPos(self,sysObj,direction):
		maxAvailShiftPos=0
		objList=[]
		sysX1=sysObj.getProperty(OB_POS_X)
		sysY1=sysObj.getProperty(OB_POS_Y)
		sysX2=sysX1+sysObj.getProperty(OB_DIMENSION_X)
		sysY2=sysY1+sysObj.getProperty(OB_DIMENSION_Y)
		tempSysX1=sysX1
		tempSysX2=sysX2
		tempSysY2=sysY2
		tempSysY1=sysY1
		xList=[]
		yList=[]
		olw=self.getProperty(OB_OUTLINE_WIDTH)
		if self.parentSystem.getProperty(OB_TYPE)=='System':
			# check if parent has any child
			if len(self.parentSystem.getObjectList())>1:
				for aKey in self.parentSystem.getObjectList():
					aParChild=self.parentSystem.theLayout.getObject(aKey)
					if aParChild.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):	
						childX1=aParChild.getProperty(OB_POS_X)
						childX2=childX1+aParChild.getProperty(OB_DIMENSION_X)
						childY1=aParChild.getProperty(OB_POS_Y)
						childY2=childY1+aParChild.getProperty(OB_DIMENSION_Y)				
						# bottom
						if direction == DIRECTION_DOWN:
							if childY1>=sysY2 :
								if childX1<sysX1 and childX2>sysX1:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1<sysX2 and sysX2<childX2:
									objList.append(aParChild)


						# right
						elif direction == DIRECTION_RIGHT: 
							if childX1>=sysX2 :
								if childY1<sysY1 and childY2>sysY1:
									objList.append(aParChild)
								elif childY1>sysY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
							
						# left
						elif direction==DIRECTION_LEFT:
							if childX2<=sysX1 :
								if sysY1>childY1 and childY2>sysY1:
									objList.append(aParChild)
								elif sysY1<childY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
									
						# top
						elif direction==DIRECTION_UP:
							if childY2<=sysY1 :
								if childX1<=sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and sysX2<childX2:
									objList.append(aParChild)
									
									
				#bottom
				if direction == DIRECTION_DOWN:
					if len(objList)>0:
						for i in range (len(objList)):
							yList.append(objList[i].getProperty(OB_POS_Y))
						yList.sort()
						y=yList[0]
					else:
							y=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)
					maxAvailShiftPos=y-sysY2-olw
				# right				
				elif direction==DIRECTION_RIGHT:
					if len(objList)>0:
						for i in range (len(objList)):
							xList.append(objList[i].getProperty(OB_POS_X))
						xList.sort()
						x=xList[0]
					else:
						x=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)
					maxAvailShiftPos=x-sysX2-olw
						
				# left
				elif direction==DIRECTION_LEFT:	
					if len(objList)>0:
						for i in range (len(objList)):
							xList.append(objList[i].getProperty(OB_POS_X)+objList[i].getProperty(OB_DIMENSION_X))
						xList.sort()
						x=xList[len(xList)-1]
					else:	
						x=olw
					maxAvailShiftPos=sysX1-x-olw

				# top
				elif direction==DIRECTION_UP:
					if len(objList)>0:
						for i in range (len(objList)):
							yList.append(objList[i].getProperty(OB_POS_Y)+objList[i].getProperty(OB_DIMENSION_Y))
						yList.sort()
						y=yList[len(yList)-1]
					else:	

						y=olw
					maxAvailShiftPos=sysY1-y-olw
					
			else:
				#parent has no child
				# bottom
				if direction == DIRECTION_DOWN:
					while (tempSysY2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
						tempSysY2=tempSysY2+1
						if (tempSysY2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
							maxAvailShiftPos=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)-sysY2-olw
							break
							
				# right
				elif direction == DIRECTION_RIGHT: 
					while (tempSysX2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
						tempSysX2=tempSysX2+1
						if (tempSysX2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
							maxAvailShiftPos=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)-sysX2-olw
							break

				# left
				elif direction==DIRECTION_LEFT:
					while (tempSysX1>4):
						tempSysX1=tempSysX1-1
						if tempSysX1<=4:
							maxAvailShiftPos=sysX1-tempSysX1-olw
							break

				# top
				elif direction==DIRECTION_UP:
					while (tempSysY1>4):
						tempSysY1=tempSysY1-1
						if tempSysY1<=4:
							maxAvailShiftPos=sysY1-tempSysY1-olw
							break
			return maxAvailShiftPos

		
	
	def getMaxDiagShiftPos(self,direction):
		dir=direction
		maxpos=[]
		objList=[]
		xmax=0
		ymax=0
		sysX1=self.getProperty(OB_POS_X)
		sysY1=self.getProperty(OB_POS_Y)
		sysX2=sysX1+self.getProperty(OB_DIMENSION_X)
		sysY2=sysY1+self.getProperty(OB_DIMENSION_Y)
		tempSysX1=sysX1
		tempSysX2=sysX2
		tempSysY2=sysY2
		tempSysY1=sysY1
		childX1=0
		childX2=0
		childY1=0
		childY2=0
		olw=self.getProperty(OB_OUTLINE_WIDTH)
		if self.parentSystem.getProperty(OB_TYPE)=='System':
			# check if parent has any child
			if len(self.parentSystem.getObjectList())>1:
				for aKey in self.parentSystem.getObjectList():
					aParChild=self.parentSystem.theLayout.getObject(aKey)
					if aParChild.getProperty(OB_FULLID)!=self.getProperty(OB_FULLID):	
						childX1=aParChild.getProperty(OB_POS_X)
						childX2=childX1+aParChild.getProperty(OB_DIMENSION_X)
						childY1=aParChild.getProperty(OB_POS_Y)
						childY2=childY1+aParChild.getProperty(OB_DIMENSION_Y)
			
						if dir==DIRECTION_BOTTOM_RIGHT:
							if childY1>sysY2 :
								if childX1<sysX1 and childX2>sysX1:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1<sysX2 and sysX2<childX2:
									objList.append(aParChild)
								elif childX1>sysX2:
 									objList.append(aParChild)
							elif childX1>sysX2 :
								if childY1<sysY1 and childY2>sysY1:
									objList.append(aParChild)
								elif childY1>sysY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
						elif dir==DIRECTION_BOTTOM_LEFT:
							if childY1>sysY2 :
								if childX1<sysX1 and childX2>sysX1:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1<sysX2 and sysX2<childX2:
									objList.append(aParChild)
								elif childX1<sysX1:
									objList.append(aParChild)
							elif childX2<sysX1 :
								if sysY1>childY1 and childY2>sysY1:
									objList.append(aParChild)
								elif sysY1<childY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
						elif dir==DIRECTION_TOP_RIGHT:
							if childY2<=sysY1 :
								if childX1<sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and sysX2<childX2:
									objList.append(aParChild)
								elif childX1>sysX2:
									objList.append(aParChild)
							elif childX1>=sysX2 :
								if childY1<sysY1 and childY2>sysY1:
									objList.append(aParChild)
								elif childY1>sysY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
						elif dir==DIRECTION_TOP_LEFT:
							if childX2<=sysX1 :
								if sysY1>childY1 and childY2>sysY1:
									objList.append(aParChild)
								elif sysY1<childY1 and childY2<sysY2:
									objList.append(aParChild)
								elif childY1<sysY2 and sysY2<childY2:
									objList.append(aParChild)
							if childY2<=sysY1 :
								if childX1<=sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and childX2<sysX2:
									objList.append(aParChild)
								elif childX1>sysX1 and sysX2<childX2:
									objList.append(aParChild)
								elif childX2<sysX2:
									objList.append(aParChild)
									

			if dir==DIRECTION_BOTTOM_RIGHT:
				if len(objList)>0:
					for o in objList:
						childX1=o.getProperty(OB_POS_X)
						childY1=o.getProperty(OB_POS_Y)	
						childX2=childX1+o.getProperty(OB_DIMENSION_X)	
						childY2=childY1+o.getProperty(OB_DIMENSION_Y)
						while not self.isOverlap(sysX1,sysY1,tempSysX2,tempSysY2,childX1,childY1,childX2,childY2):
							tempSysX2+=1
							tempSysY2+=1
						maxpos.append(min(tempSysX2-sysX2,tempSysY2-sysY2))
						tempSysX2=sysX2
						tempSysY2=sysY2
						
				else:
					while (tempSysY2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
						tempSysY2+=1
						if (tempSysY2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
							ymax=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)-sysY2-olw
							break
					while (tempSysX2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
						tempSysX2+=1
						if (tempSysX2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
							xmax=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)-sysX2-olw
							break
					maxpos.append(min(xmax,ymax))

			elif dir==DIRECTION_BOTTOM_LEFT:
				if len(objList)>0:
					for o in objList:
						childX1=o.getProperty(OB_POS_X)
						childY1=o.getProperty(OB_POS_Y)	
						childX2=childX1+o.getProperty(OB_DIMENSION_X)	
						childY2=childY1+o.getProperty(OB_DIMENSION_Y)
						while not self.isOverlap(childX1,childY1,childX2,childY2,tempSysX1,sysY1,sysX2,tempSysY2):
							tempSysX1-=1
							tempSysY2+=1
						maxpos.append(min(sysX1-tempSysX1,tempSysY2-sysY2))
						tempSysX1=sysX1
						tempSysY2=sysY2
				else:
					while (tempSysX1>4):
						tempSysX1=tempSysX1-1
						if tempSysX1<=4:
							xmax=sysX1-tempSysX1-olw
							break
					while (tempSysY2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
						tempSysY2+=1
						if (tempSysY2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
							ymax=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)-sysY2-olw
							break
					maxpos.append(min(xmax,ymax))

			elif dir==DIRECTION_TOP_RIGHT:
				if len(objList)>0:
					for o in objList:
						childX1=o.getProperty(OB_POS_X)
						childY1=o.getProperty(OB_POS_Y)	
						childX2=childX1+o.getProperty(OB_DIMENSION_X)	
						childY2=childY1+o.getProperty(OB_DIMENSION_Y)
                                              
						while not self.isOverlap(childX1,childY1,childX2,childY2,sysX1,tempSysY1,tempSysX2,sysY2):
							tempSysX2+=1
							tempSysY1-=1
						maxpos.append(min(tempSysX2-sysX2,sysY1-tempSysY1))
						tempSysX2=sysX2
						tempSysY1=sysY1
					
				else:
					while (tempSysY2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
						tempSysY2+=1
						if (tempSysY2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
							ymax=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)-sysY2-olw
							break
					while (tempSysX2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
						tempSysX2+=1
						if (tempSysX2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)):
							xmax=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_X)-sysX2-olw
					maxpos.append(min(xmax,ymax))

			elif dir==DIRECTION_TOP_LEFT:
				if len(objList)>0:
					for o in objList:
						childX1=o.getProperty(OB_POS_X)
						childY1=o.getProperty(OB_POS_Y)	
						childX2=childX1+o.getProperty(OB_DIMENSION_X)	
						childY2=childY1+o.getProperty(OB_DIMENSION_Y)
						while not self.isOverlap(childX1,childY1,childX2,childY2,tempSysX1,tempSysY1,sysX2,sysY2):
							tempSysX1-=1
							tempSysY1-=1
						maxpos.append(min(sysY1-tempSysY1,sysX1-tempSysX1))
						tempSysX1=sysX1
						tempSysY1=sysY1
				else:
					while (tempSysX1>4):
						tempSysX1=tempSysX1-1
						if tempSysX1<=4:
							xmax=(sysX1-tempSysX1-olw)
							break
					while (tempSysY2<self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
						tempSysY2+=1
						if (tempSysY2>=self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)):
							ymax=(self.parentSystem.getProperty(SY_INSIDE_DIMENSION_Y)-sysY2-olw)
							break
					maxpos.append(min(xmax,ymax))
		
		maxpos.sort()	
		return maxpos[0]

	def getMaxDiagShiftNeg(self,direction):
		dir=direction
		xmaxneg=0
		ymaxneg=0
		x1List=[]
		y1List=[]
		x2List=[]
		y2List=[]
		olw=self.getProperty(OB_OUTLINE_WIDTH)
		if self.getProperty(OB_TYPE)=='System':
			sysX1=0
			sysY1=0
			sysX2=sysX1+self.getProperty(OB_DIMENSION_X)
			sysY2=sysY1+self.getProperty(OB_DIMENSION_Y)
			textWidth=self.getProperty(OB_SHAPEDESCRIPTORLIST).getRequiredWidth()		
			# check if it has any child
			if len(self.getObjectList())>0:
				for aKey in self.getObjectList():
					aChildObj=self.theLayout.getObject(aKey)
					childX1=aChildObj.getProperty(OB_POS_X)
					childX2=childX1+aChildObj.getProperty(OB_DIMENSION_X)
					childY1=aChildObj.getProperty(OB_POS_Y)
					childY2=childY1+aChildObj.getProperty(OB_DIMENSION_Y)+olw*8
					x1List.append(childX1)
					y1List.append(childY1)
					x2List.append(childX2)
					y2List.append(childY2)
				x1List.sort()
				y1List.sort()
				x2List.sort()
				y2List.sort()
			
			if dir==DIRECTION_BOTTOM_RIGHT or dir == DIRECTION_BOTTOM_LEFT:
				if len(y2List)>0:	
					ymaxneg=(sysY2-(y2List[len(y2List)-1]))	
				else:
					ymaxneg=(sysY2-sysY1)-olw*10	
				if len(x2List)>0:
					if textWidth>x2List[len(x2List)-1]:
						xmaxneg=(sysX2-textWidth)		
					else:
						xmaxneg=sysX2-x2List[len(x2List)-1]
				else:
					xmaxneg=sysX2-textWidth
			
			elif dir==DIRECTION_TOP_RIGHT or dir == DIRECTION_TOP_LEFT:
				if len(y1List)>0:	
					ymaxneg=(sysY2-(y2List[len(y2List)-1]))	
				else:
					ymaxneg=(sysY2-sysY1)-olw*10	
				if len(x2List)>0:
					if textWidth>x2List[len(x2List)-1]:
						xmaxneg=(sysX2-textWidth)		
					else:
						xmaxneg=sysX2-x2List[len(x2List)-1]
				else:
					xmaxneg=sysX2-textWidth
			
		return -xmaxneg,-ymaxneg

	def isOverlap(self,x1,y1,x2,y2,childX1,childY1,childX2,childY2):
		olw = self.getProperty( OB_OUTLINE_WIDTH )
		isOverlap=True
		objX1=x1
		objY1=y1
		objX2=x2
		objY2=y2
#

		if objY1>=childY2:
			isOverlap=False
		elif objY2<=childY1:
			isOverlap=False
		elif  objY1<=childY1 or objY1<=childY2:
			if objX2<childX1:
				isOverlap=False
			elif objX1>childX2:
				isOverlap=False
			else:
				isOverlap=True
#		
		return isOverlap

	def getMaxDimensions(self,x,y):
		objX1=x
		objY1=y
		objX2=x+SYS_MINWIDTH
		objY2=y+SYS_MINHEIGHT
		maxAvailWidth=SYS_MINWIDTH
		maxAvailHeight=SYS_MINHEIGHT
		if len(self.getObjectList())>0:
			for aKey in self.getObjectList(): 
				aChildObj=self.theLayout.getObject(aKey)
				childX1=aChildObj.getProperty(OB_POS_X)
				childY1=aChildObj.getProperty(OB_POS_Y)
				childX2=childX1+aChildObj.getProperty(OB_DIMENSION_X)
				childY2=childY1+aChildObj.getProperty(OB_DIMENSION_Y)
				if self.isOverlap(objX1,objY1,objX2,objY2,childX1,childY1,childX2,childY2):
					maxAvailWidth=childX1-objX1-10				
					maxAvailHeight=childY1-objY1-10
					break
		return (maxAvailWidth,maxAvailHeight)
		
	def isWithinParent(self,x,y,type,child=None):
		olw = self.getProperty( OB_OUTLINE_WIDTH )
		parX1=0
		parY1=0
		parX2= parX1+self.getProperty(OB_DIMENSION_X)
		parY2= parY1+self.getProperty(OB_DIMENSION_Y)
		childX1=x
		childY1=y
		childX2=0
		childY2=0
		if type==OB_TYPE_SYSTEM:
			childX2=childX1+SYS_MINWIDTH
			childY2=childY1+SYS_MINHEIGHT
		elif type==OB_TYPE_VARIABLE:
			childX2=x+VAR_MINWIDTH
			childY2=y+VAR_MINHEIGHT
		elif type==OB_TYPE_PROCESS:
			childX2=x+PRO_MINWIDTH
			childY2=y+PRO_MINHEIGHT
		elif type==OB_TYPE_TEXT:
			childX2=x+TEXT_MINWIDTH
			childY2=y+TEXT_MINHEIGHT
		if not child == None:
			if childX2<x+child.getProperty(OB_DIMENSION_X):
				childX2=x+child.getProperty(OB_DIMENSION_X)
			if childY2<y+child.getProperty(OB_DIMENSION_Y):
				childY2=y+child.getProperty(OB_DIMENSION_Y)
		childY2+=olw*8 #height of the parent label
		if ((parX1<childX1 and parY1<childY1) and (parX2>childX2 and parY2>childY2)):
			return True
		else:
			return False

	def isOverlapSystem(self,x,y,type,anobj=None):
		isOverlapSystem=True
		objX1=x
		objY1=y	
		if type==OB_TYPE_VARIABLE:
			objX2=x+VAR_MINWIDTH
			objY2=y+VAR_MINHEIGHT
		elif type==OB_TYPE_PROCESS:
			objX2=x+PRO_MINWIDTH
			objY2=y+PRO_MINHEIGHT
		elif type==OB_TYPE_TEXT:
			objX2=x+TEXT_MINWIDTH
			objY2=y+TEXT_MINHEIGHT
		if not anobj == None:
			if objX2<x+anobj.getProperty(OB_DIMENSION_X):
				objX2=x+anobj.getProperty(OB_DIMENSION_X)
			if objY2<y+anobj.getProperty(OB_DIMENSION_Y):
				objY2=y+anobj.getProperty(OB_DIMENSION_Y)
		if len(self.getObjectList())>0:
			for aKey in self.getObjectList(): 
				aChildObj=self.theLayout.getObject(aKey)
				if aChildObj.getProperty(OB_TYPE)=='System':
					childX1=aChildObj.getProperty(OB_POS_X)
					childY1=aChildObj.getProperty(OB_POS_Y)
					childX2=childX1+aChildObj.getProperty(OB_DIMENSION_X)
					childY2=childY1+aChildObj.getProperty(OB_DIMENSION_Y)
					if self.isOverlap(objX1,objY1,objX2,objY2,childX1,childY1,childX2,childY2):
						isOverlapSystem=True
						break
					else:
						isOverlapSystem=False
				else:
					isOverlapSystem=False
		else:
			isOverlapSystem=False
		return isOverlapSystem
	

	

	def getGraphUtils( self ):
		return self.theLayout.graphUtils()


	def buttonReleased( self ):
		pass


	
