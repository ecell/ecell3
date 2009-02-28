//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
//
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
//

#ifndef __LIBECS_WIN32_ERRNO_H
#define __LIBECS_WIN32_ERRNO_H

#define _EPERM   1
#define _ENOENT  2
#define _EBADF   9
#define _ENOMEM 12
#define _EACCES 13
#define _EEXIST 17
#define _EINVAL 22
#define _ENOSPC 28
#define _EUKNWN -1

#ifdef __cplusplus
extern "C" {
#endif

extern int* libecs_win32_errno();

#define __libecs_errno (*libecs_win32_errno())

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* __LIBECS_WIN32_ERRNO_H */
