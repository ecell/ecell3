#!/bin/env python

import os,sys,shutil,popen2

print '''
################################################################# 
#                                                               #
# This script generates sample input files of ga.esm            #
# The difference of each model is initial concentration.        #
# This script run E-Cell using same parameters Km=12.0 and      # 
# KcF=4.0, and save time-course data under each directory.      #
#                                                               #
# The following files are required.                             #
# (1, 2, or 3 is applied to * )                                 #
#                                                               #
#  simple*.em     - one of model file                           #
#  obserbable.py  - ess file                                    #
#                                                               #
# The following files and directories are generated.            #
#                                                               #
#  simple*.eml    - EML file converted from simple*.em          #
#  Data*%sS.ecd    - observable time-course of simple*.em        #
#  Data*%sP.ecd    - observable time-course of simple*.em        #
#                                                               #
# [Note] If the path of ecell3-em2eml and ecell3-session is not #
#       currect, this script could not be executed correctly.   #
#                                                               #
################################################################# 
''' %( os.sep, os.sep )

if os.path.isdir('Data1'):
	print "deleting Data1..."
	shutil.rmtree('Data1')

if os.path.isdir('Data2'):
	print "deleting Data2..."
	shutil.rmtree('Data2')

if os.path.isdir('Data3'):
	print "deleting Data3..."
	shutil.rmtree('Data3')

print "creating Data1..."
os.mkdir('Data1')
print "creating Data2..."
os.mkdir('Data2')
print "creating Data3..."
os.mkdir('Data3')

if os.name == "nt":
    anEm2eml = 'ecell3 ecell3-em2eml'
    anEcell3session = 'ecell3 ecell3-session'
else:
    anEm2eml = 'ecell3-em2eml'
    anEcell3session = 'ecell3-session'

aCommand = anEm2eml + ' simple1.em'
print aCommand
os.system(aCommand)

aCommand = anEm2eml + ' simple2.em'
print aCommand
os.system(aCommand)

aCommand = anEm2eml + ' simple3.em'
print aCommand
os.system(aCommand)

aCommand = anEcell3session + ' -e observable.py --parameters=\"{\'_EML_\':\'simple1.eml\',\'_KmS_\':12.0,\'_KcF_\':4.0,\'_Data_\':\'Data1\'}\"'
print aCommand
os.system(aCommand)

aCommand = anEcell3session + ' -e observable.py --parameters=\"{\'_EML_\':\'simple2.eml\',\'_KmS_\':12.0,\'_KcF_\':4.0,\'_Data_\':\'Data2\'}\"'
print aCommand
os.system(aCommand)

aCommand = anEcell3session + ' -e observable.py --parameters=\"{\'_EML_\':\'simple3.eml\',\'_KmS_\':12.0,\'_KcF_\':4.0,\'_Data_\':\'Data3\'}\"'
print aCommand
os.system(aCommand)

# end of this script
