import ecell.Window
import gtk

class FullPNQueue:

    def __init__( self, anAttachmentPoint, aFullPNList ):
        self.backwardQueue = []
        self.forwardQueue = []
        self.theRawFullPNList = aFullPNList
        aWindow = ecell.Window.Window( self.__class__.__name__ + '.glade', "hbox1" )
        aWindow.openWindow()
        aFrame = aWindow['hbox1']
        self.backButton = aWindow['backbutton']
        self.forwardButton = aWindow['forwardbutton']
        anAttachmentPoint.add( aFrame )
        self.backButton.connect( "clicked", self.__goBack )
        self.forwardButton.connect( "clicked", self.__goForward )
        self.callbackList = []
        self.__updateNavigatorButtons()

    def registerCallback( self, aFunction ):
        self.callbackList.append( aFunction )
        apply( aFunction, [self.theRawFullPNList] )


    def pushFullPNList( self, aRawFullPNList ):
        self.backwardQueue.append( self.__copyList ( self.theRawFullPNList  ) )
        self.forwardQueue = []

        self.__applyFullPNList( aRawFullPNList )
        self.__updateNavigatorButtons()
        
    def getActualFullPNList( self ):
        return self.__copyList( self.theRawFullPNList )

    def __applyFullPNList( self, aRawFullPNList ):
        self.theRawFullPNList = self.__copyList( aRawFullPNList )
        for aFunction in self.callbackList:

            apply( aFunction, [aRawFullPNList] )


    def __copyList( self, aList ):
        newList = []
        for anItem in aList:
            if type( anItem ) in [type( [] ) , type( () ) ]:
                newList.append( anItem )
            else:
                newList.append( self.__copyList( anItem ) )
        return newList


        
    def __goBack(self, *args):
        if len( self.backwardQueue ) == 0:
            return
        rawFullPNList = self.backwardQueue.pop()
        self.forwardQueue.append( self.__copyList( self.theRawFullPNList ) )
        self.__applyFullPNList( rawFullPNList )
        self.__updateNavigatorButtons()

        
    def __goForward( self, *args ):
        if len( self.forwardQueue ) == 0:
            return
        rawFullPNList = self.forwardQueue.pop()
        self.backwardQueue.append( self.__copyList( self.theRawFullPNList ) )
        self.__applyFullPNList( rawFullPNList )
        self.__updateNavigatorButtons()
        

    def __updateNavigatorButtons( self ):
        if len( self.backwardQueue ) == 0:
            backFlag = gtk.FALSE
        else:
            backFlag = gtk.TRUE
        if len( self.forwardQueue ) == 0:
            forFlag = gtk.FALSE
        else:
            forFlag = gtk.TRUE
        self.forwardButton.set_sensitive( forFlag )
        self.backButton.set_sensitive( backFlag )
        

# read buttons fromfile
# create from EntityList
# create from ropertyWindow
# selection functions in propwindow
# treewalker
# search
