#
# This is a sample script file of EMS ( E-Cell Session Manager script )
# for Globus
#

import os

setEnvironment('Globus2')  

setTmpDirRemovable(False)  # not delete tmp directory

setConcurrency(2) # set concurrency

# -------------------------------
# set up SystemProxy's properties

aSystemProxy = getSystemProxy()

aSystemProxy.setPassword('your_passwd')  # password for globus gate-keeper
aSystemProxy.gridProxyInit()      # initialize gate-keeper

# create a list of hosts on which jobs are conducted.
aHostList = ['host1.domain',
             'host2.domain']
aSystemProxy.setHosts(aHostList) # set the host list

# -------------------------------

MODEL_FILE = 'model.eml'
ESS_FILE = 'runsession.py'

# Register jobs.

aJobIDList = []

for i in xrange(0,2):
	
	VALUE_OF_S = i * 1000
	aParameterDict = { 'MODEL_FILE': MODEL_FILE, 'VALUE_OF_S': VALUE_OF_S }

	#registerEcellSession( ESS file, parameters, files that ESS uses )
	aJobID = registerEcellSession( ESS_FILE, aParameterDict, [ MODEL_FILE, ])
	aJobIDList.append( aJobID ) # Memorize the job IDs in aJobIDList.

# Run the registered jobs.

run()

for aJobID in aJobIDList: 

	print " --- job id = %s ---" %aJobID
	print getStdout( aJobID )  # Print the output of each job. 

