from Constants import *
from __future__ import division
import Numeric as nu
import gnome.canvas

class LineDescriptor:

    def __init__( self, graphUtils, parentObject ):
        self.theGraphUtils = graphUtils
        self.parentObject = parentObject
        label = self.parentObject.getProperty( CO_NAME )
        firstArrow = self.parentObject.getProperty( CO_HASARROW1 )
        secondArrow = self.parentObject.getProperty( CO_HASARROW2 )
        linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
        self.dirConversion=[ -1,0, 1,-1,2,2,2,-1,3,3,3 ] 
        self.matrixArray = [ nu.array([[1,0,-1],[0,1,0]]), nu.array([[1,0,1],[0,1,0]]),nu.array([[1,0,0],[0,1,-1]]) ,nu.array([[1,0,0],[0,1,1]]) ] 
        self.Qarray = [1,2,4,3] 
        self.theDescriptorList = {\
         "textbox": [ "textbox",CV_TEXT, SD_FILL, SD_FILL, 0, [ label, 0,0,0,0 ], {}  ],\
        ARROWHEAD1: [ARROWHEAD1,CV_LINE, SD_ARROWHEAD, SD_FILL, 1, [ 0,0,0,0, firstArrow, gtk.FALSE, linewidth ], {} ],\
        ARROWHEAD2: [ ARROWHEAD2,CV_LINE, SD_ARROWHEAD, SD_FILL, 1, [ 0,0,0,0 ,secondArrow, gtk.FALSE, linewidth], {} ] }

    def renameLabel( self, newLabel ):
        aDescriptor = self.theDescriptorList[ "textbox"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        aSpecific[0] = newLabel


    def getDescriptorList( self ):
        return self.theDescriptorList

    def getDescriptor ( self, shapeName ):
        if shapeName in self.theDescriptorList.keys():
            aDescriptor =  self.theDescriptorList[shapeName]
            return aDescriptor
        raise Exception (" Shape %s doesnot exist")
        
    def reCalculate( self ):
        self.calculateEndpoints()
        self.calculateSpecific()
        self.calculateTextPosition()


    def calculateTextPosition( self):
        pass

    def __checkQ(self):
        code= ( self.x2< self.x1)*2+( self.y2> self.y1)
        return self.Qarray[code]

    def calculateEndpoints( self ):
        ( self.x1, self.y1 ) = self.parentObject.getProperty( CO_ENDPOINT1 )
        ( self.x2, self.y2 ) = self.parentObject.getProperty( CO_ENDPOINT2 )

        self.dir1 = self.dirConversion[self.parentObject.getProperty( CO_DIRECTION1 )]
        self.dir2 = self.dirConversion[self.parentObject.getProperty( CO_DIRECTION2 )]
        self.Q = self.__checkQ()

        self.endpointmatrix1 = nu.array( [self.x1, self.y1,ARROWHEAD_LENGTH ] )
        self.endpointmatrix2 = nu.array( [self.x2, self.y2,ARROWHEAD_LENGTH ] )
        #reshape
        self.endpointmatrix1  = nu.reshape(self.endpointmatrix1,(3,1))
        self.endpointmatrix2 = nu.reshape(self.endpointmatrix2,(3,1))
    
        # get matrix for x1,y1,x2,y2
        m1 = self.matrixArray[ self.dir1 ] #cheCk: yields a multi-dimensional array of arrays
        m1= nu.reshape(m1,(2,3)) #cheCk: join and return the 3rd and 4th aray element combined  
        m2 = self.matrixArray[self.dir2]
        m2=nu.reshape(m2,(2,3))
        
        #cheCk: insidematrix1 for ARROWHEAD1, connection of AH with MS
        #cheCk: numeric.dot is a dot multiplication process function to yield matrix    
        self.insidematrix1 = nu.dot(m1, self.endpointmatrix1)
        #cheCk: insidematrix2 for ARROWHEAD2, connection of AH with MS
        self.insidematrix2 = nu.dot(m2, self.endpointmatrix2)

        (self.insidex1, self.insidey1) = (self.insidematrix1[0][0], self.insidematrix1[1][0])
        (self.insidex2, self.insidey2) = (self.insidematrix2[0][0], self.insidematrix2[1][0])
        firstArrow = self.parentObject.getProperty( CO_HASARROW1 )
        secondArrow = self.parentObject.getProperty( CO_HASARROW2 )
    
        #cheCk: ARROWHEAD1 is the default arrow Line
        aDescriptor = self.theDescriptorList[ ARROWHEAD1]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        #cheCk: ARROWHEAD1: [ARROWHEAD1,CV_LINE, SD_ARROWHEAD, SD_FILL, 1, [ 0,0,0,0, firstArrow, gtk.FALSE, linewidth ], {} ]
        aSpecific [0] = self.x1 # coords in charge of arrowhead connection with entities
        aSpecific [1] = self.y1
        aSpecific [2] = self.insidex1 # coords in charge of arrowhead connection with 'ms'
        aSpecific [3] = self.insidey1
        aSpecific [4] = firstArrow #true or false i guess

        #cheCk: ARROWHEAD2 is the default non arrow Line
        aDescriptor = self.theDescriptorList[ ARROWHEAD2]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        aSpecific [0] = self.x2 
        aSpecific [1] = self.y2
        aSpecific [2] = self.insidex2
        aSpecific [3] = self.insidey2
        aSpecific [4] = secondArrow
        
            

class multiBcurveLineSD(LineDescriptor):
    def __init__(self, parentObject, graphUtils ):

        LineDescriptor.__init__( self, graphUtils, parentObject)    
        linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
        self.previousEP = ""
        self.previousCA = ""
        self.theDescriptorList["MultiBezierCurve"] = ["MultiBezierCurve", CV_BPATH, SD_MOVINGLINE, SD_FILL, 1, [[0,0,0],linewidth-2],{} ]
        
        # CREATE INITIAL CONTROL POINTS
        
        self.calculateInitialCoefMatrix()
        self.reCalculate()

    def calculateInitialCoefMatrix(self ):
        # inital coefmatrix and endpoints
        self.totalBcurveCount = 3
        self.coefArray = [ 0,0, 0.25, 0.2 , 0.15 , 0.1 ,   0.3,0.3,\
			   0.4, 0.35, 0.50,  0.50,  0.6, 0.6,\
			   0.7, 0.75, 0.85, 0.92, 1 , 1]
        self.parentObject.thePropertyMap[CO_CONTROL_POINTS] = self.coefArray


    def calculateSpecific(self):
        self.coefArray = self.parentObject.getProperty( CO_CONTROL_POINTS )
        # compare enpoint and coefmatrix with previos values
        endpointhash = str( [self.insidex1, self.insidey1, self.insidex2, self.insidey2] )
        coefhash = str( self.coefArray )
        if endpointhash == self.previousEP and coefhash == self.previousCA:
            return
        x0 = self.insidex1
        y0 = self.insidey1
        xn = self.insidex2
        yn = self.insidey2
        dx=xn-x0
        dy=yn-y0
        self.previousEP = endpointhash
        self.previousCA = coefhash
        # calculate pointmatrix
        self.totalBcurveCount = int(( len( self.coefArray ) - 2 ) / 6)
        coefPairCount = len( self.coefArray)/2
        cM = nu.transpose( nu.reshape( self.coefArray, ( coefPairCount, 2 ) ) )
        dM = nu.array( [ [ dx,0],[0,dy] ] )
        pointMatrix = nu.dot( dM, cM )

        pointMatrix += [[x0],[y0]] 

        
        # calculate descriptor specific
        aDescriptorList = []
        for i in range(0, self.totalBcurveCount ):
            coordVector = list(nu.ravel(nu.transpose( pointMatrix[0:2, i*3+1:i*3+4]) ))
            forwardPath = tuple( [ gnome.canvas.CURVETO ] + coordVector )

            coordVector =list(nu.transpose( pointMatrix[0:2, i*3:i*3+3] ))
            coordVector.reverse()
            reverseVector = []
            for aVect in coordVector:
                reverseVector.extend( list(aVect) )

            reversePath = tuple( [ gnome.canvas.CURVETO ] + reverseVector )
            aDescriptorList.insert(i, reversePath)
            aDescriptorList.insert(i, forwardPath)
        aDescriptorList.insert(0, ( gnome.canvas.MOVETO_OPEN, x0, y0) )
        self.theDescriptorList["MultiBezierCurve"][SD_SPECIFIC] [BPATH_PATHDEF] = aDescriptorList
    
    
    def  calculateTextPosition( self):
        aDescriptor = self.theDescriptorList["textbox"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        label = self.parentObject.getProperty( CO_NAME )
        aSpecific[ TEXT_TEXT ] = label
        (aSpecific [ TEXT_ABSX ], aSpecific [ TEXT_ABSY ]) = self.__getTextPosition()

    
    def __getTextPosition(self):
        middleBpath = str(int( self.totalBcurveCount / 2 ) + 1)
        aDescriptor = self.theDescriptorList["MultiBezierCurve"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        x1,y1,x2,y2 = aSpecific [BPATH_PATHDEF][1][1:5]
        if x2-x1 !=0:
            m = (y2 -y1)/(x2 -x1)
        else:
            m=1
        c = y1 - m*x1
        y = y1 - (y1-y2)/2
        if m != 0:
            x=(y-c)/m
        else: 
            x = x1
        return x,y
       
        
        

class curvedLineSD(LineDescriptor ):
    def __init__( self, parentObject, graphUtils ):

        LineDescriptor.__init__( self, graphUtils, parentObject )
        #NAME, TYPE, FUNCTION, COLOR, Z, POINTS, PROPERTIES 
        linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
        #"textbox": [ "textbox",CV_TEXT, SD_FILL, SD_FILL, 0, [ label, 0,0,0,0 ], {}  ]
        self.theDescriptorList["curvedLine"] = ["curvedLine",CV_BPATH, SD_MOVINGLINE, SD_FILL, 1,[[0,0,0],linewidth-2],{}]


        self.coef = 0.85
        #about coef
        #THIS IS FOR THE 4 ATTACH POINTS of process, how each arrowhead will look like when they are joined to any of them    
        #1st two ... right arrowhead towards entity, left arrowhead towards entity
        ###############################################################################
        #1st array defines curvature, 2nd array defines endpoints connecting to arrowhead of each attach point
        ###############################################################################
        self.Matrix = [ nu.array([[1,0,-self.coef,0],[0,1,0,0]]), nu.array([[1,0,self.coef,0],[0,1,0,0]]), 
               nu.array([[1,0,0,0],[0,1,0,-self.coef]]),  nu.array([[1,0,0,0],[0,1,0,self.coef]])]

        self.reCalculate()
        

    def calculateSpecific( self ):
        # calculate inside line
        aDescriptor = self.theDescriptorList["curvedLine"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        x0,y0 =  self.insidex1, self.insidey1 #cheCk: for ARROWHEAD1 or 2 i think..
        x3,y3 =  self.insidex2, self.insidey2 #cheCk: for ARROWHEAD2
        #cheCk: Difference in the 2 end points? that will be the length in between
        dy=y3-y0
        dx=x3-x0
        if dy<0:
            dy =-dy
        if dx<0:
            dx=-dx
        
        x1,y1 = self.__getXYPos(self.dir1,x0,y0,dx,dy) #parse in inside ARROWHEAD1 x y
        x2,y2 = self.__getXYPos(self.dir2,x3,y3,dx,dy) #parse in inside ARROWHEAD2 x y
        
        #These are the coords of the the Bpath in the middle!
        #1st defines the coords of Bpath, connecting to ARROWHEAD1 insidex and insdiey
        #2nd and 3rd Sets the initial curvature of the Bpaths, strangely .. there are 2 Bpaths
        aSpecific [BPATH_PATHDEF] = [(gnome.canvas.MOVETO_OPEN, x0,y0), 
(gnome.canvas.CURVETO, x1,y1,x2,y2,x3,y3), (gnome.canvas.CURVETO,
x2,y2,x1-1,y1-1,x0-1,y0-1)]


    #I think this is "standard"        
    def __getXYPos(self,dir,x,y,dx,dy):
        a = nu.reshape(self.Matrix[dir],(2,4))

        b = nu.array([x, y,dx,dy])
        b= nu.reshape(b,(4,1))
        cMatrix = nu.dot(a,b)

        return (cMatrix[0][0], cMatrix[1][0])
        
            
            
    def __getCurvedType(self,Q,d1,d2):
        code=(Q-1)*16+d1*40+d2
        return self.codeMatrix[code]
           
        
    def calculateTextPosition( self):
        aDescriptor = self.theDescriptorList["textbox"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        label = self.parentObject.getProperty( CO_NAME )
        aSpecific[ TEXT_TEXT ] = label
        (aSpecific [ TEXT_ABSX ], aSpecific [ TEXT_ABSY ]) = self.__getTextPosition()

    
    def __getTextPosition(self):
        aDescriptor = self.theDescriptorList["curvedLine"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        x1,y1,x2,y2 = aSpecific [BPATH_PATHDEF][1][1:5]
        if x2-x1 !=0:
            m = (y2 -y1)/(x2 -x1)
        else:
            m=1
        c = y1 - m*x1
        y = y1 - (y1-y2)/2
        if m != 0:
            x=(y-c)/m
        else: 
            x = x1
        return x,y
        
        
        
class corneredLineSD(LineDescriptor ):
    def __init__( self, parentObject, graphUtils ):

        LineDescriptor.__init__( self, graphUtils, parentObject )
        #NAME, TYPE, FUNCTION, COLOR, Z, POINTS, PROPERTIES 
        linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
        # self.theDescriptorList["curvedLine"] = ["curvedLine",CV_BPATH, SD_MOVINGLINE, SD_FILL, 1,[[0,0,0],linewidth-2],{}]
        self.theDescriptorList["lineL"] = [ "lineL", CV_LINE, SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ]
        self.theDescriptorList["lineC"] = [ "lineC", CV_LINE,  SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ]
        self.theDescriptorList["lineR"] = [ "lineR", CV_LINE,  SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth],{} ] 
        '''
        self.Matrix = {'Bpath1':[ nu.array([[1,0,-self.coef,0],[0,1,0,0]]), 
                                nu.array([[1,0,self.coef,0],[0,1,0,0]]), 
                                nu.array([[1,0,0,0],[0,1,0,-self.coef]]),  
                                nu.array([[1,0,0,0],[0,1,0,self.coef]])] }
        '''
        
        self.matrixMap = {'lineL':[nu.array([[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,0,0,1,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0]]), 
                       nu.array([[1,0,0,0,-1,0],[0,1,0,0,0,0],[1,0,0,0,-1,0],[0,0,0,1,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,-1]])], 
                  'lineC':[nu.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,1,0,0,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]]), 
                       nu.array([[1,0,0,0,-1,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,1,0,0,0,-1],[0,0,1,0,0,0],[0,1,0,0,0,-1]])], 
                  'lineR':[nu.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,0,0,0,0],[0,1,0,0,0,0]]), 
                       nu.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[1,0,0,0,-1,0],[0,1,0,0,0,0]]), 
                       nu.array([[0,0,1,0,0,0],[0,1,0,0,0,-1],[0,0,1,0,0,0],[0,0,0,1,0,0]])]}  
        
        self.codeMatrix = [ 1,3,1,1,1,0,1,0,1,0,1,0,0,0,2,0,1,3,1,1,1,0,0,1,0,0,0,2,1,0,0,1,0,1,0,1,3,1,1,1,0,0,0,2,0,1,0,1,0,1,1,0,3,1,1,1,0,1,1,0,0,0,2,0]   
        
        self.reCalculate()

    def calculateSpecific( self ):
        aProcessID = self.parentObject.getProperty( CO_PROCESS_ATTACHED ) 
        aProcessObj = self.parentObject.theLayout.getObject( aProcessID )
        self.aProWidth = aProcessObj.getProperty(OB_DIMENSION_X)
        self.aProHeight = aProcessObj.getProperty(OB_DIMENSION_Y)
    
        #get Quadrant
        if self.Q in [3,4]:
            self.aProWidth=-self.aProWidth
        if self.Q in [1,4]:
            self.aProHeight = -self.aProHeight
        
        self.specMatrix =  nu.array( [self.insidex1, self.insidey1,self.insidex2, self.insidey2,self.aProWidth, self.aProHeight  ] )
        self.specMatrix = nu.reshape(self.specMatrix,(6,1))

        type = self.__getLineType(self.Q, self.dir1, self.dir2)

        
        # get inside line
        for aKey in self.theDescriptorList.keys():
            if aKey in ('lineL','lineR','lineC'):
                aDescriptor = self.theDescriptorList[aKey]
                aSpecific = aDescriptor[ SD_SPECIFIC ]
                m = self.matrixMap[aDescriptor[ SD_NAME ]][type]
                m =  nu.reshape(m ,(4,6))
                
                self.specPointMtx = nu.dot(m,self.specMatrix)
                for i in range (len(self.specPointMtx)):
                    aSpecific[i]=self.specPointMtx[i][0]
                
    def calculateTextPosition( self):
        aDescriptor = self.theDescriptorList["textbox"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        label = self.parentObject.getProperty( CO_NAME )
        aSpecific[ TEXT_TEXT ] = label
        aSpecific [ TEXT_ABSX ], aSpecific [ TEXT_ABSY ]  =self.__getTextposition()
    
    def __getLineType(self, Q, d1, d2):
        code=(Q-1)*16+d1*4+d2
        return self.codeMatrix[code]
                            
    
    def __getTextposition( self):
        aDescriptor = self.theDescriptorList["lineC"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        textX, textY = aSpecific[0], aSpecific[1]
        textX = textX-(textX- aSpecific[2])/2
        return (textX, textY)
    
                

class StraightLineSD( LineDescriptor ):

    def __init__( self, parentObject, graphUtils ):
        LineDescriptor.__init__( self, graphUtils, parentObject )
        #NAME, TYPE, FUNCTION, COLOR, Z, POINTS, PROPERTIES 
        linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
        self.theDescriptorList["straightline"]= [ "straightline", CV_LINE, SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ]
        
        self.reCalculate()


    def calculateSpecific( self ):
        # calculate inside line:
        aDescriptor = self.theDescriptorList["straightline"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        aSpecific [0] = self.insidex1
        aSpecific [1] = self.insidey1
        aSpecific [2] = self.insidex2
        aSpecific [3] = self.insidey2
        
    def calculateTextPosition( self):
        aDescriptor = self.theDescriptorList["textbox"]
        aSpecific = aDescriptor[ SD_SPECIFIC ]
        label = self.parentObject.getProperty( CO_NAME )
        aSpecific[ TEXT_TEXT ] = label
        if self.insidex2 -self.insidex1 !=0:
            m = (self.insidey2 -self.insidey1)/(self.insidex2 -self.insidex1)
        else:
            m=1
        c = self.insidey1 - m*self.insidex1
        y = self.insidey1 - (self.insidey1-self.insidey2)/2
        if m != 0:
            x=(y-c)/m
        else: 
            x = self.insidex1
        aSpecific [ TEXT_ABSX ], aSpecific [ TEXT_ABSY ] = x,y
        
        
                    
