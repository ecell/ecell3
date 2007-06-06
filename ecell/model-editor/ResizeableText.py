#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
import gtk.gdk
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas

class ResizeableText:
    def __init__( self, aRoot, aCanvas, x ,y , fillColor, text, anchor):
        self.theRoot=aRoot
        self.thePathwayCanvas=aCanvas
        self.theText=text
        self.x=x
        self.y=y
        self.anchor=anchor
        self.fill=fillColor
        self.ppu=0
        #self.parentID=parentID
    
        self.theCanvasText=self.theRoot.add( gnomecanvas.CanvasText, x=self.x, y=self.y, fill_color_gdk = self.fill, text = self.theText, anchor = self.anchor, font="Sans 9" )

        # register self to Pathway canvas   
        self.thePathwayCanvas.registerText(self, self.theText)
        self.resizeText()


    def resizeText(self):
        self.ppu=self.thePathwayCanvas.getZoomRatio()
        pgfd = self.theCanvasText.get_property("font-desc").copy()
        pangosize =1024*self.ppu*9
        pgfd.set_size( int(pangosize) )
        self.theCanvasText.set_property('font-desc', pgfd )

    def renameText(self, newText):
        self.thePathwayCanvas.setLabelText(self,newText)
        self.theText=newText

    def destroy(self):
        #deregister at pathway canvas
        self.thePathwayCanvas.deregisterText(self)
        self.theCanvasText.destroy()

    def w2i(self,x,y):
        return self.theCanvasText.w2i(x,y)


    def set_property(self, aProperty, aValue=None):
        if aProperty=='size':
            self.resizeText()
        else:
            self.theCanvasText.set_property(aProperty, aValue )
            if aProperty=='text':
                self.renameText(aValue) 

    def get_property(self,aProperty):
        return self.theCanvasText.get_property(aProperty)


    def connect(self,*args):
        """
        args[0]=event name
        args[1]=callback method
        args[2]=name
        """
        eventName=args[0]
        callback=args[1]
        shapeName=args[2]
        self.theCanvasText.connect(eventName,callback,shapeName)

    def move(self,deltax,deltay):
        self.theCanvasText.move(deltax,deltay)

    
