import gnome.canvas
import gtk.gdk

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
    
        self.theCanvasText=self.theRoot.add( gnome.canvas.CanvasText, x=self.x, y=self.y, fill_color_gdk = self.fill, text = self.theText, anchor = self.anchor )

        # register self to Pathway canvas   
        self.thePathwayCanvas.registerText(self, self.theText)
        self.resizeText()


    def resizeText(self):
        self.ppu=self.thePathwayCanvas.getZoomRatio()
        pgfd = self.theCanvasText.get_property("font-desc").copy()
        pangosize =1024*self.ppu*10
        pgfd.set_size( pangosize )
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

    
