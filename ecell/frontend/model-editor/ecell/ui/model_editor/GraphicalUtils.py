#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
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

import gtk
import numpy as n
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas

from ecell.ui.model_editor.Constants import *

class GraphicalUtils:
    def __init__( self, aGtkWindow ):
        self.theGtkWindow = gtk.Window()
        self.theGdkWindow = self.theGtkWindow.window
        self.theColorMap = self.theGtkWindow.get_colormap()
        self.theStyle = self.theGtkWindow.get_style()
        self.testCanvas = gnomecanvas.Canvas()
        self.testCanvasRoot = self.testCanvas.root()
        self.testText=self.testCanvasRoot.add(gnomecanvas.CanvasText, text = 'Hello', font="Sans 9" )

        self.opMatrix= n.array(((0,0,1,0,-1,0,0,0),(-1,0,0,0,0,0,1,0),(0,0,0,1,0,-1,0,0),(0,-1,0,0,0,0,0,1)))

        #for overlapping
        self.o1=n.array([[0,0,-1,0],[1,0,0,0],[0,0,0,-1],[0,1,0,0]])
        self.o2=n.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,-1]])

        #for within
        self.o3=n.array([[-1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
        self.o4=n.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,-1]])

        #for resizing (x1,y1,x2,y2)
        self.resizeMap= {}

        self.mapPos={DIRECTION_LEFT:n.array([[-1,0],[0,0],[0,0],[0,0]]), 
                 DIRECTION_RIGHT:n.array([[0,0],[0,0],[1,0],[0,0]]),
                 DIRECTION_UP:n.array([[0,0],[0,-1],[0,0],[0,0]]), 
                 DIRECTION_DOWN:n.array([[0,0],[0,0],[0,0],[0,1]]),
                 DIRECTION_BOTTOM_RIGHT:n.array([[0,0],[0,0],[1,0],[0,1]]), 
                 DIRECTION_BOTTOM_LEFT:n.array([[-1,0],[0,0],[0,0],[0,1]]),
                 DIRECTION_TOP_RIGHT:n.array([[0,0],[0,-1],[1,0],[0,0]]), 
                 DIRECTION_TOP_LEFT:n.array([[-1,0],[0,-1],[0,0],[0,0]])}

        self.mapNeg={DIRECTION_LEFT:n.array([[1,0],[0,0],[0,0],[0,0]]), 
                 DIRECTION_RIGHT:n.array([[0,0],[0,0],[-1,0],[0,0]]),
                 DIRECTION_UP:n.array([[0,0],[0,1],[0,0],[0,0]]), 
                 DIRECTION_DOWN:n.array([[0,0],[0,0],[0,0],[0,-1]]),
                 DIRECTION_BOTTOM_RIGHT:n.array([[0,0],[0,0],[-1,0],[0,-1]]), 
                 DIRECTION_BOTTOM_LEFT:n.array([[1,0],[0,0],[0,0],[0,-1]]),
                 DIRECTION_TOP_RIGHT:n.array([[0,0],[0,1],[-1,0],[0,0]]), 
                 DIRECTION_TOP_LEFT:n.array([[1,0],[0,1],[0,0],[0,0]])}

        

        #each point (x1,y1,x2,y2):[x1,y1,x2,y2,posx,posy,negx,negy]
        self.twoRectMap={DIRECTION_BOTTOM_RIGHT:n.array([[[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,-1],[0,0,1,0,1,0,0,0],[0,0,0,1,0,1,0,0]],[[0,0,1,0,0,0,-1,0],[0,1,0,0,0,0,0,0],[0,0,1,0,1,0,0,0],[0,0,0,1,0,1,0,0]]]),\
                 DIRECTION_BOTTOM_LEFT:n.array([[[1,0,0,0,-1,0,0,0],[0,0,0,1,0,0,0,-1],[0,0,1,0,0,0,0,0],[0,0,0,1,0,1,0,0]],[[1,0,0,0,-1,0,0,0],[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,1,0],[0,0,0,1,0,1,0,0]]]),\
                 DIRECTION_TOP_RIGHT:n.array([[[1,0,0,0,0,0,0,0],[0,1,0,0,0,-1,0,0],[0,0,1,0,1,0,0,0],[0,1,0,0,0,0,0,1]],[[0,0,1,0,0,0,-1,0],[0,1,0,0,0,-1,0,0],[0,0,1,0,1,0,0,0],[0,0,0,1,0,0,0,0]]]),\
                 DIRECTION_TOP_LEFT:n.array([[[1,0,0,0,-1,0,0,0],[0,1,0,0,0,-1,0,0],[0,0,1,0,0,0,0,0],[0,1,0,0,0,0,0,1]],[[1,0,0,0,-1,0,0,0],[0,1,0,0,0,-1,0,0],[1,0,0,0,0,0,1,0],[0,0,0,1,0,0,0,0]]])}

        self.dotRectMap={DIRECTION_LEFT:[0], DIRECTION_RIGHT:[2],DIRECTION_UP:[1],DIRECTION_DOWN:[3],   
                 DIRECTION_BOTTOM_RIGHT:[2,3],DIRECTION_BOTTOM_LEFT:[0,3],DIRECTION_TOP_RIGHT:[2,1],    
                                 DIRECTION_TOP_LEFT:[0,1]}

    def getGdkColorByName( self, aColorName ):
        return self.theColorMap.alloc_color( aColorName )

    def getGdkColorByRGB( self, rgbArray ):
        aGdkColor = self.theStyle.white.copy()
        aGdkColor.red = rgbArray[0]
        aGdkColor.green = rgbArray[1]
        aGdkColor.blue = rgbArray[2]
        return aGdkColor

    def getRRGByName( self, aColorName ):
        gdkColor = self.getGdkColorByName( aColorName )

        return [ gdkColor.red, gdkColor.green, gdkColor.blue ]

    def getTextDimensions( self, aText ):
        self.testText.set_property('text',aText)
        width = self.testText.get_property('text-width')
        height = self.testText.get_property('text-height')
        return (height, width )

    def truncateTextToSize ( self, aText, aSize ):
        return aText
        
    def truncateLabel(self,aLabel,lblWidth,limit,minlbllength):
        cutlength=int(lblWidth-limit)
        if cutlength>minlbllength:
            truncated=aLabel[0:minlbllength-3]+'...'
            return truncated
        cutlength+=2
        truncated=aLabel[:-cutlength]+'...' 
        return truncated

    def areOverlapping( self, rect1, rect2 ):
        b= n.concatenate( (rect1,rect2))
        return  n.sometrue(n.less(n.dot(self.opMatrix,b),0))

    def calcOverlap(self,r1,rn):
        if rn==None:
            return False
        rm=n.dot(self.o1,r1)
        rn=n.dot(self.o2,rn)
        eq=rm+rn
        bm=n.greater(eq,0)
        bv=n.sometrue(bm,0)
        return not n.alltrue(bv)
        
    def calcWithin(self,r1,rn):
        rm=n.dot(self.o3,r1)
        rn=n.dot(self.o4,rn)
        eq=rm+rn
        bm=n.greater(eq,0)
        bv=n.alltrue(bm,0)
        return n.alltrue(bv)    

    def calcMaxShiftPos(self,r1,rn,dir,rpar):
        #rm=n.array([r1[0],r1[1],r1[2],r1[3]])
        #prevrm=n.array([r1[0],r1[1],r1[2],r1[3]])
        rm=n.reshape(r1,(4,1))
        #prevrm=n.reshape(r1,(4,1))
        rpar=n.reshape(rpar,(4,1))
        rxy=self.getBigIterator(rm,rpar,dir)
        ex = max( abs(rxy[0]), abs(rxy[2]) )
        ey = max( abs(rxy[1]), abs(rxy[3]) )
        rxy = n.reshape( rxy, (4,1) )
        minint=0.0
        maxint=1.0
        m=0
        stepx=1;stepy=1
        if ex!=0:
            stepx=(maxint/ex)
        if ey!=0:
            stepy=(maxint/ey)
        #if ex==0:
        #    stepx=stepy
        #if ey==0:
        #    stepy=stepx
        #form rxy
        #rxy=n.array([ex,ey])
        #rxy=n.reshape(rxy,(2,1))
        #rxy=n.dot(self.mapPos[dir],rxy)
        
        if rn==None:
            rm=rm+rxy
            rm=n.reshape(rm,(4,))
            return n.around(rm)
        rmtemp = n.array( rm )
        while (maxint-minint)>stepx or (maxint-minint)>stepy:
            #prevrm[0][0]=rm[0][0]
            #prevrm[1][0]=rm[1][0]
            #prevrm[2][0]=rm[2][0]
            #prevrm[3][0]=rm[3][0]
            prevrm = n.array( rm )
            m=(minint+maxint)/2
            rxytemp=rxy*m
            rmtemp=rm+rxytemp

            cond = not self.calcOverlap(rmtemp,rn)
            
            if cond:
                minint=m
            else:
                maxint=m

        rm=n.reshape(rmtemp,(4,))
        return n.around(rm)

    def someGreater(self,amatrix):
        bm=n.greater(amatrix,0)
        bv=n.sometrue(bm,0)
        return n.alltrue(bv)

    def someSmaller(self,amatrix):
        bm=n.less(amatrix,0)
        bv=n.sometrue(bm,0)
        return n.alltrue(bv)

    def allGreater(self,amatrix):
        bm=n.greater(amatrix,0)
        bv=n.alltrue(bm,0)
        return n.alltrue(bv)
    
    def allSmaller(self,amatrix):
        bm=n.less(amatrix,0)
        bv=n.alltrue(bm,0)
        return n.alltrue(bv)

    def getBigIterator(self,r1,rpar,dir):
        x=r1[0][0];y=r1[1][0];x2=r1[2][0];y2=r1[3][0]
        px=rpar[0][0];py=rpar[1][0];px2=rpar[2][0];py2=rpar[3][0]
        anArray = n.array( [0,0,0,0] )
        if dir&DIRECTION_UP==DIRECTION_UP:
            anArray[1] = -y + 1
        if dir&DIRECTION_DOWN==DIRECTION_DOWN:
            anArray[3] = py2 - y2 - 1
        if dir&DIRECTION_LEFT==DIRECTION_LEFT:
            anArray[0] = -x + 1
        if dir&DIRECTION_RIGHT==DIRECTION_RIGHT:
            anArray[2] = px2 - x2 - 1
        return anArray
        
    def calcMaxShiftNeg(self,r1,rn,dir):
        rm=n.array([r1[0],r1[1],r1[2],r1[3]])
        rm=n.reshape(rm,(4,1))
        ex,ey=self.getSmallIterator(rm,rn,dir)
        rxy=n.array([ex,ey])
        rxy=n.reshape(rxy,(2,1))
        rxy=n.dot(self.mapNeg[dir],rxy)
        rm=rm+rxy
        rm=n.reshape(rm,(4,))
        return n.around(rm)     

    def getSmallIterator(self,r1,rn,dir):
        mex=r1[0][0]
        mey=r1[1][0]
        mex2=r1[2][0]-mex
        mey2=r1[3][0]-mey
        mex=0;mey=0
        #sort childxy
        cx=n.sort(rn[0]);smallcx=cx[0]
        cy=n.sort(rn[1]);smallcy=cy[0]
        cx2=n.sort(rn[2]);index=n.argmax(cx2);largecx2=cx2[index]
        cy2=n.sort(rn[3]);index=n.argmax(cy2);largecy2=cy2[index]
        availx=0;availy=0;ex=0;exy=0;eyx=0;ey=0
        
        if dir==DIRECTION_UP or dir==DIRECTION_DOWN:
            return 0,mey2-largecy2
        if dir==DIRECTION_RIGHT or dir==DIRECTION_LEFT :
            return mex2-largecx2,0
        if dir==DIRECTION_BOTTOM_RIGHT or dir==DIRECTION_BOTTOM_LEFT:
            ex=mex2-largecx2
            ey=mey2-largecy2
        elif dir==DIRECTION_TOP_RIGHT or dir==DIRECTION_TOP_LEFT:
            ex=mex2-largecx2
            ey=mey2-largecy2
        
        exy=(ex*mey2)/mex2
        eyx=(ey*mex2)/mey2
        if exy<ey:
            availx=ex
            availy=exy
        else:
            availx=eyx
            availy=ey
        return availx,availy

    def buildTwoRect(self,x1,y1,x2,y2,posx,posy,negx,negy,dir):
        negx=abs(negx);negy=abs(negy)
        a=n.array([x1,y1,x2,y2,posx,posy,negx,negy])
        a=n.reshape(a,(8,1))
        r=n.dot(self.twoRectMap[dir],a)
        twoRect=n.array([[r[0][0][0],r[0][1][0],r[0][2][0],r[0][3][0]],[r[1][0][0],r[1][1][0],r[1][2][0],r[1][3][0]]])
        return n.around(twoRect)

    def getRectDotxy(self,x1,y1,x2,y2,dir):
        dotRect=n.array([x1,y1,x2,y2])
        posx=self.dotRectMap[dir][0]
        posy=self.dotRectMap[dir][1]
        return dotRect[posx],dotRect[posy]
        
