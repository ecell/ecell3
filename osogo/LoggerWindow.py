#!/usr/bin/env python

from Window import *
from gtk import *
from Numeric import *

from ecssupport import *

class LoggerWindow(Window):

    def __init__( self, aMainWindow ):

        Window.__init__( self, 'LoggerWindow.glade' )
        
        self.theMainWindow = aMainWindow
        self.theFullPNList = self.theMainWindow.theDriver.getLoggerList()
        self.theEntryList = self['clist3']
        self.theEntryList.connect( 'select_row', self.selectPropertyName )
        self.theList = []
        self.initialize()
        self.addHandlers( {'on_button121_clicked':self.saveAllData} )
        self.addHandlers( {'on_exit_activate' : self.closeWindow} )
        self.theSelectedFullPNList = self.theFullPNList


    def initialize( self ):

        self.setEntryList()
        

    def saveAllData( self, obj ):

        if self["checkbutton26"].get_active():
               interval = self["spinbutton8"].get_text()
               print interval

        k=0

        for fpn in self.theSelectedFullPNList :
            aLogger = self.theMainWindow.theDriver.getLogger( fpn )
            start= aLogger.getStartTime()
            end  = aLogger.getEndTime()

            if self["checkbutton25"].get_active():
               start= end = self["spinbutton7"].get_text()
            data = aLogger.getLoggerData(start,end,interval=0)

            for i in xrange(len(data)):
                test = data[i]
                print test[0],test[1],start,end
                x = str(test[0]) 
                y = str(test[1])
                afpnlist = self.theMainWindow.theDriver.getLoggerList()
                j=afpnlist[k][2]
                print j
                output = open('%s.ecd'%j,'a')
                output.writelines("%s\t%s\n" %(x,y))
            k=k+1


    def selectPropertyName( self, aClist, row, column, event_obj ):

        self.theSelectedFullPNList = []

        if  self["radiobutton18"].get_active():
            for aRowNumber in aClist.selection:
                aPropertyName = aClist.get_text(aRowNumber,0)   
                self.theSelectedFullPNList.append(aPropertyName)
        else:
            self.theSelectedFullPNList = self.theFullPNList


    def setEntryList( self ):

        for fpn in self.theFullPNList :
            theFullPN = createFullPNString( fpn )
            aLogger = self.theMainWindow.theDriver.getLogger( fpn )
            start = aLogger.getStartTime()
            end = aLogger.getEndTime()
            aList = [ theFullPN, start, end ]
            aList = map( str, aList )
            self.theList.append( aList )

        self.update()


    def update( self ):

        self.theEntryList.clear()
        for aValue in self.theList:
            self.theEntryList.append( aValue )


    def closeWindow ( self, obj ):

        gtk.mainquit()


### test code

if __name__ == "__main__":

    class MainWindow:

        def __init__( self ):
            self.theSimulator = simulator()

    class simulator:

        def __init__( self ):

            self.dic={('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

        def getProperty( self, fpn ):
            return self.dic[fpn]

        def setProperty( self, fpn, value ):
            self.dic[fpn] = value

        def getLogger( self, fpn ):
            logger = Logger( fpn )
            return logger

        def getLoggerList( self ):
            fpnlist = ((SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity'),
                       (SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity'),
                       (SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity'))
            return fpnlist

    class Logger:

        def __init__( self, fpn ):
            pass

        def getStartTime( self ):
            return 0

        def getEndTime( self ):
            return 100

        def getLoggerData( self ,start=0,end=0,interval=0):
            return array([[0,0],[0.1,0.1],[0.2,0.3],[0.3,0.7],[0.4,0.9],[0.5,1.0]])
              
    def mainQuit( obj, data ):
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aMainWindow = MainWindow()
        aLoggerWindow = LoggerWindow( aMainWindow )
        mainLoop()

    main()
