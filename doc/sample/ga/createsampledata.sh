#!/bin/sh

cat << EOT
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
#  Data*/S.ecd    - observable time-course of simple*.em        #
#  Data*/P.ecd    - observable time-course of simple*.em        #
#                                                               #
# [Note] If the path of ecell3-em2eml and ecell3-session is not #
#       currect, this script could not be executed correctly.   #
#                                                               #
################################################################# 

EOT

if [ -d 'Data1' ] ; then
	rm -rf Data1
fi

if [ -d 'Data2' ] ; then
	rm -rf Data2
fi

if [ -d 'Data3' ] ; then
	rm -rf Data3
fi

mkdir Data1
mkdir Data2
mkdir Data3

ecell3-em2eml simple1.em
ecell3-em2eml simple2.em
ecell3-em2eml simple3.em
ecell3-session -e observable.py --parameters="{'_EML_':'simple1.eml','_KmS_':12.0,'_KcF_':4.0,'_Data_':'Data1'}"
ecell3-session -e observable.py --parameters="{'_EML_':'simple2.eml','_KmS_':12.0,'_KcF_':4.0,'_Data_':'Data2'}"
ecell3-session -e observable.py --parameters="{'_EML_':'simple3.eml','_KmS_':12.0,'_KcF_':4.0,'_Data_':'Data3'}"

# end of this script
