# sample of testrun

from ecell.ECDDataFile import *
import os

print "onerun.ess --------------------------------> start "

# ---------------------------------------------------------
# load the model
# ---------------------------------------------------------
loadModel( _EML_ )


# ---------------------------------------------------------
# set parameter
# ---------------------------------------------------------
anEntity = createEntityStub( 'Process:/:E' )
anEntity.setProperty( 'KmS', _KmS_ )
anEntity.setProperty( 'KcF', _KcF_ )


# ---------------------------------------------------------
# create logger stubs
# ---------------------------------------------------------
S_Logger = createLoggerStub( 'Variable:/:S:Value' )
S_Logger.create()
S = createEntityStub( 'Variable:/:S' )
P_Logger = createLoggerStub( 'Variable:/:P:Value' )
P_Logger.create()
P = createEntityStub( 'Variable:/:P' )

# ---------------------------------------------------------
# run
# ---------------------------------------------------------
# print some values
message( 'KmS= \t%s' %anEntity.getProperty('KmS')  )
message( 'KcF= \t%s' %anEntity.getProperty('KcF')  )
message( '\n' )

message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'P:Value= \t%s' % P.getProperty( 'Value' ) )
# run
duration = 1000
message( '\n' )
message( 'run %s sec.\n' % duration )
run( duration )
# print results
message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'P:Value= \t%s' % P.getProperty( 'Value' ) )
message( '\n' )


# -----------------------------------
# save simulated time-course
# -----------------------------------
aS = ECDDataFile( S_Logger.getData(0,1000,10) )
aP = ECDDataFile( P_Logger.getData(0,1000,10) )

message('saving S.ecd..')
aS.setDataName( S_Logger.getName() )
aS.setNote( 'observable S' )
aS.save( _Data_ + os.sep + 'S.ecd' )

message('saving P.ecd..')
aP.setDataName( P_Logger.getName() )
aP.setNote( 'observable P' )
aP.save( _Data_ + os.sep + 'P.ecd' )



print "onerun.ess --------------------------------> end "



