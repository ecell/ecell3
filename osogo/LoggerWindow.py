#!/usr/bin/env python

from Window import *
from gtk import *
from Numeric import *

from ecell.ecssupport import *

class LoggerWindow(Window):

    def __init__( self, session, aMainWindow ):

        Window.__init__( self, 'LoggerWindow.glade' )
        
        self.theSession = session
        self.theEntryList = self['loggerWindow_clist']
        self.theEntryList.connect( 'select_row', self.selectPropertyNameByClick )
     #   self.theEntryList.connect( 'unselect_row', self.selectPropertyNameByClick )
        self.theList = []
        self.initialize()
        self.addHandlers( { 'on_Now_button_clicked' : self.saveAllData, 
                            'on_exit_activate' : self.closeWindow } )
        self.theMainWindow = aMainWindow
        

    def initialize( self ):

        self.update()
        

    def saveAllData( self, obj):
        aSelectedFullPNList = self.selectPropertyName()
        
        if self["every_checkbutton"].get_active():
               interval = self["every_spinbutton"].get_text()

        for fullPNString in aSelectedFullPNList :
            fullpn=createFullPN(fullPNString)
            aLogger = self.theMainWindow.theDriver.getLogger( fullpn )
            start= aLogger.getStartTime()
            end  = aLogger.getEndTime()

            if self["at_checkbutton"].get_active():
               start= end = self["at_spinbutton"].get_text()
     #      The code that currently implemented by Logger is the one of the bottom.    
     #      data = aLogger.getData(start,end,interval = None )
            data = aLogger.getData()
            filename=string.split(fullPNString,'/')
            filename=filename[-1]
            filename=string.split(filename,':')
            filename=filename[0] + '_' + filename[1] + '_' + filename [-1]
            output= open("%s.ecd"%filename,  "w")
         
            for i in xrange(len(data)):
                test = data[i]
                x = str(test[0]) 
                y = str(test[1])
  
                output.writelines("%s\t%s\n" %(x,y))

            output.close()

    def selectPropertyName( self ):
        aCList = self['loggerWindow_clist']
        selectedFullPNList = []

        if  self["Selected_radiobutton"].get_active():
            for aRowNumber in aCList.selection:
                   
                aPropertyName = aCList.get_text(aRowNumber,0)
                
                selectedFullPNList.append( aPropertyName )
        else:
                selectedFullPNList = self.theFullPNList

                
    

        return  selectedFullPNList

    def selectPropertyNameByClick( self, aCList, row, column, event_obj ):
        return self.selectPropertyName()

    def update( self ):

        self.theFullPNList = self.theSession.getLoggerList()

        self.theEntryList.clear()
        self.theList = []

        for aFullPNString in self.theFullPNList :
            aFullPN = createFullPN( aFullPNString )
            aLogger = self.theSession.getLogger( aFullPN )
            start = str( aLogger.getStartTime() )
            if self.theSession.theRunningFlag:
                end = 'running'
            else:
                end = str( aLogger.getEndTime() )
            aList = [ aFullPNString, start, end ]
            self.theList.append( aList )

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
