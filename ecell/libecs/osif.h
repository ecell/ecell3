//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//	This file is a part of E-CELL2.
//	Original codes of E-CELL1 core were written by Koichi TAKAHASHI
//	<shafi@e-cell.org>.
//	Some codes of E-CELL2 core are minor changed from E-CELL1
//	by Naota ISHIKAWA <naota@mag.keio.ac.jp>.
//	Other codes of E-CELL2 core and all of E-CELL2 UIMAN are newly
//	written by Naota ISHIKAWA.
//	All codes of E-CELL2 GUI are written by
//	Mitsui Knowledge Industry Co., Ltd. <http://bio.mki.co.jp/>
//
//	Latest version is availabe on <http://bioinformatics.org/>
//	and/or <http://www.e-cell.org/>.
//END_V2_HEADER
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 *	$Id$
 :	$Log$
 :	Revision 1.2  2005/11/19 09:23:59  shafi
 :	Kouichi -> Koichi
 :
 :	Revision 1.1  2002/04/30 11:21:53  shafi
 :	gabor's vvector logger patch + modifications by shafi
 :	
 :	Revision 1.5  2002/01/15 14:53:43  ishikawa
 :	osif_add_path()
 :	
 :	Revision 1.4  2001/10/21 15:27:12  ishikawa
 :	osif_is_dir()
 :	
 :	Revision 1.3  2001/03/23 18:51:17  naota
 :	comment for credit
 :	
 :	Revision 1.2  2001/01/13 01:31:47  naota
 :	Can be compiled by VC, but does not run.
 :
 :	Revision 1.1  2000/12/30 15:09:46  naota
 :	Initial revision
 :
//END_RCS_HEADER
 *::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */
/*
 * OS dependent module
 * Written by ISHIKAWA Naota.  (C) 2000 Keio university
 */
#ifdef __cplusplus
extern "C" {
#endif
#ifndef __OSIF_H__
#define __OSIF_H__ 1

#ifdef WIN32
#include "unistd.h"
#else
#include <sys/types.h>
#endif


long osif_get_pid();
off_t osif_disk_free(const char *__path);	/* by K Bytes */
int osif_mkdir(const char *__name);
int osif_is_dir(const char *__name);
int osif_load_dll(const char *__name);
int osif_add_path(const char *__path, int to_first);

#ifdef __cplusplus
} /* end of extern "C" */
#endif /* __cplusplus */
#endif /* __OSIF_H__ */
