#!/usr/bin/env python

from gtk import *
from gtkextra import *
from Window import *
from Numeric import *
import GDK
import GTK
import re
import string
from PluginWindow import *
from ecssupport import *
#from testTracerglade import *
#from data1 import *

class TracerWindow( PluginWindow ):

    def __init__( self, dirname, data, pluginmanager, root=None ):

	PluginWindow.__init__( self, dirname, data, pluginmanager, root )

        IDflag = 1
        if len( self.theFullPNList() ) > 1:
            for aFullID in self.theFullIDList():
                if aFullID == self.theFullID():
                    IDflag = 0
                else:
                    IDflag = 1
        
        if IDflag == 1:
            if self.theDriver.isNumber(self.theFullPN()):
                self.openWindow()
                self.thePluginManager.appendInstance( self )                    
                PluginWindow.initialize( self, root )
                self.initialize()
            else:
                aFullPNString = createFullPNString( self.theFullPN() )
                self.theSession.printMessage( "%s: not numerical data\n" % aFullPNString )                    
                
        else:
            aClassName = self.__class__.__name__
            for aFullPN in self.theFullPNList():
                if self.theDriver.isNumber( aFullPN ):
                    a = self.thePluginManager.createInstance( aClassName, (aFullPN,), root)
                else:
                    aFullPNString = createFullPNString( self.theFullPN() )     
                    self.theSession.printMessage( "%s: not numerical data\n" % aFullPNString )                    
            

    def initialize( self ):
        self.xaxis = None
        self.yaxis = None
        self.arg = 0
        self.step_size = 0
        self.theLoggerList = []
        self.theDataList = []

        self['toolbar1'].set_style(GTK.TOOLBAR_ICONS)
        self.addHandlers( { 'on_button9_clicked' : self.popInputWindow,
                            'checkbutton1_clicked' : self.changeScale } )
        
        aWindowWidget = self.getWidget( 'frame8' )
        self.theColorMap = aWindowWidget.get_colormap()

        self.canvas = GtkPlotCanvas(450,300)
        self.canvas.set_background( self.theColorMap.alloc("light blue") )

        self.plot = GtkPlot(width=0.8, height=0.8)
        self.plot.set_background( self.theColorMap.alloc("white") )
        self.plot.set_yscale(PLOT_SCALE_LINEAR)
        self.plot.autoscale()
        self.plot.axis_set_ticks(PLOT_AXIS_X, 1, 1)
        self.plot.axis_set_visible(PLOT_AXIS_RIGHT, FALSE)
        self.plot.axis_set_visible(PLOT_AXIS_TOP, FALSE)
        self.plot.grids_set_visible(TRUE, FALSE, TRUE, FALSE)
        self.plot.major_hgrid_set_attributes( PLOT_LINE_SOLID, 1.0,
                                              self.theColorMap.alloc("light blue") )
        self.plot.major_vgrid_set_attributes( PLOT_LINE_SOLID, 1.0,
                                              self.theColorMap.alloc("light blue") )
        self.plot.axis_hide_title(PLOT_AXIS_TOP)
        self.plot.axis_hide_title(PLOT_AXIS_RIGHT)
        self.plot.axis_hide_title(PLOT_AXIS_LEFT)
        self.plot.axis_hide_title(PLOT_AXIS_BOTTOM)                
        self.plot.hide_legends()
        self.canvas.add_plot(self.plot, 0.16, 0.05)
        self.theDataList = []
        for num in [2,3,4,5,6,7,8]:
            data = GtkPlotData()
            data.set_symbol(PLOT_SYMBOL_NONE, PLOT_SYMBOL_OPAQUE, 0, 0, self.getColor(num-2))
            data.set_line_attributes(PLOT_LINE_SOLID, 1, self.getColor(num-2))
        
            self.plot.add_data(data)
            self.theDataList.append(data)

        self['frame8'].add(self.canvas)

        self.createLogger()

        self.plot.clip_data(TRUE)

        aWindowWidget.show_all()
        self.update()


    def popInputWindow(self, obj):

	Window.__init__(self, 'plugins/InputWindow.glade',None)

        self.addHandlers({'entry1_activate' : self.changexaxis,
                          'entry2_activate' : self.changeyaxis,
                          'entry3_activate' : self.changeyaxis,
                          'entry4_activate' : self.changeyaxis,
                          'entry5_activate' : self.changeyaxis,
                          'entry6_activate' : self.changeyaxis,
                          'entry7_activate' : self.changeyaxis,
                          'entry8_activate' : self.changeyaxis,
                          })

#        self['togglebutton2'].set_mode(TRUE)
        for num in [1,2,3,4,5,6,7]:
            self['toolbar%i'%num].set_style(GTK.TOOLBAR_ICONS)

        self['entry1'].set_text( 'time' )
        num = 1
        for fpn in self.theFullPNList():
            entrynum = num + 1
            self['entry%i'%entrynum].set_text( 'X%i'%num )
            num += 1

            
    def changeScale( self, obj ):

        if obj.get_active():
            self.plot.set_yscale(PLOT_SCALE_LOG10)
            self.plot.autoscale()
            
        else:
            self.plot.set_yscale(PLOT_SCALE_LINEAR)
            self.plot.autoscale()


    def createLogger( self ):

        n = 1
        for fpn in self.theFullPNList():
            ID = fpn[2]
            self.theLoggerList.append(self.theSession.getLogger(fpn))
            label = "label%d"%(n)
            self[label].set_text(ID)
            n += 1
        while n <= 7:
            self['button%i'%n].set_mode(TRUE)
            n += 1

        self.thePluginManager.updateLoggerWindow()
        

    def getColor( self, num ):

        aColorList = ["red", "blue", "orange", "green",
                      "purple", "navy", "brown", "black",
                      "white", "light_blue", "light_yellow", "black"]

        return self.theColorMap.alloc( aColorList[num] )
    

    def updateLoggerDataList(self):
        self.LoggerDataList =[]
        for aLogger in self.theLoggerList:
            self.LoggerDataList.append( aLogger.getData() )
            
        if self.LoggerDataList[0] != ():
            self.datanumber = array(self.LoggerDataList[0]).shape[0]
            self.step_size = int(self.datanumber/500)
        num = 0
        self.entry = {}
        while num < len(self.theLoggerList):
            if self.LoggerDataList[num] == ():
                pass
            else:
                self.LoggerDataList[num] = array( self.LoggerDataList[num])
                self.entry['X%s' % str(num+1)] = self.LoggerDataList[num][:,1][::self.step_size]
            num += 1


    def update(self):
        if self.arg % 10 == 0:
            stateList = []
            if self['togglebutton2'] == None:
                stateList = [0,0,0,0,0,0,0]
            else:
                for togglebuttonnum in [2,3,4,5,6,7,8]:
                    togglebutton = 'togglebutton%i'%togglebuttonnum
                    stateList.append(self[togglebutton].get_active())
            self.updateLoggerDataList()


            if self.LoggerDataList[0] == ():
                pass
            else:
                if self.xaxis == None:
                    x = array(self.LoggerDataList)[0][:,0][::self.step_size]
                elif self.xaxis == 'time':
                    x = array(self.LoggerDataList)[0][:,0][::self.step_size]
                else:
                    x = eval(self.xaxis,self.entry)

            yList = []
            
            if self.yaxis == None:
                num = 0
                for LoggerData in self.LoggerDataList:
                    if LoggerData == ():
                        pass
                    else:
                        if stateList[num] == 1:
                            self.theDataList[num].set_points( None, None )
                        else:
                            self.theDataList[num].set_points( x, array(LoggerData)[:,1][::self.step_size] )
                    num += 1

            else:
                num = 0
                for yaxis in self.yaxis:
                    if stateList[num] == 1:
                        self.theDataList[num].set_points(None,None)
                    elif yaxis == '':
                        self.theDataList[num].set_points(None,None)
                    else :
                        self.theDataList[num].set_points(x,eval(yaxis,self.entry))
                    num += 1
            
            self.plot.autoscale()
            self.canvas.paint()
            self.canvas.refresh()
            
        self.arg += 1
        return TRUE

    
    def changexaxis(self, obj):

        self.xaxis = obj.get_text()
        
        
    def changeyaxis(self, obj):

        self.yaxis = []
        for num in [2,3,4,5,6,7,8]:
            self.yaxis.append(self['entry%i'%num].get_text())



if __name__ == '__main__':
    class simulator :

        def __init__( self ):
            self.dic = {('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

        def getProperty( self, fpn ):
            return self.dic[fpn]

        def setProperty( self, fpn, value ):
            self.dic[fpn] = value

        def getLogger( self, fpn ):
            logger= Logger( fpn )
            return logger

        def getLoggerList( self ):
            fpnlist = ((SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity'),
                       (SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity'))
                       
            return fpnlist

    fpnlist = (('Substance','/CELL/CYTOPLASM','ATP',''),
               ('Substance','/CELL/CYTOPLASM','ADP',''))


    class Logger:

        def __init__( self, fpn ):
            if(fpn ==('Substance', '/CELL/CYTOPLASM', 'ATP', '')) :
                a = 'CYTOPLASM-Cln2_C.ecd'
                alist = []
                fp = open(a,'r')

                for line in fp.readlines():
                    if re.search('^\d+.\d+\s',line):
                        alist+=[[string.atof(re.split('\s+',line)[0]),string.atof(re.split('\s+',line)[1])]]
            elif(fpn ==('Substance', '/CELL/CYTOPLASM', 'ADP', '')) :
                a = 'CYTOPLASM-mass_C.ecd'
                alist = []
                fp = open(a,'r')

                for line in fp.readlines():
                    if re.search('^\d+.\d+\s',line):
                        alist+=[[string.atof(re.split('\s+',line)[0]),string.atof(re.split('\s+',line)[1])]]
            else:
                print "error"
            self.Data = array(alist)
            
        def getStartTime( self ):
            return 0

        def getEndTime( self ):
            return 100

        def getLoggerData( self, start, end):
            return self.Data[start:end]
        

               
    def mainQuit( obj, data ):
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        atestTracer = TracerWindow(simulator,fpnlist)
        mainloop()
#        mainLoop()

    main() 





