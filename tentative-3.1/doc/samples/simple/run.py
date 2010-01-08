
# load the model
loadModel( 'simple.eml' )

# create stubs
S_Logger = createLoggerStub( 'Variable:/:S:Value' )
S_Logger.create()
S = createEntityStub( 'Variable:/:S' )


# print some values
message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'S:MolarConc= \t%s' % S.getProperty( 'MolarConc' ) )
# run
duration = 1000
message( '\n' )
message( 'run %s sec.\n' % duration )
run( duration )


# print results
message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'S:MolarConc= \t%s' % S.getProperty( 'MolarConc' ) )

message( '\n' )

from ecell.ECDDataFile import *

message('saving S.ecd..')
aDataFile = ECDDataFile( S_Logger.getData(0,2000,.5) )
aDataFile.setDataName( S_Logger.getName() )
aDataFile.setNote( '' )
aDataFile.save( 'S.ecd' )

#message('loading')
#aNewFile = ECDDataFile()
#aNewFile.load( 'S.ecd' )
#print aNewFile.getData()[:10]
