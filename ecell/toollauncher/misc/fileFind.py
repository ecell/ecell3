#!/usr/bin/python
#
# fileFind.py
#

import fnmatch
import os
import traceback

_prune = ['(*)']

# ==========================================================================
def fileFind(dir = os.curdir):
	list = []
	names = os.listdir(dir)
	names.sort()
	for name in names:
		if name in (os.curdir, os.pardir):
			continue
		fullname = os.path.join(dir, name)

		if os.path.isdir(fullname):
			list.append(fullname+os.sep)
		else:
			list.append(fullname)

		if os.path.isdir(fullname) and not os.path.islink(fullname):
			for p in _prune:
				if fnmatch.fnmatch(name, p):
					break
			else:
				list = list + fileFind(fullname)
	return list

# end of fileFild