//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

#ifndef _LIBECS_WIN32_DL_COMPAT_H
#define _LIBECS_WIN32_DL_COMPAT_H

#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int libecs_win32_dl_init(void);
extern int libecs_win32_dl_exit(void);
extern const char *libecs_win32_dl_error(void);
extern const char *libecs_win32_dl_get_search_path(void);
extern int libecs_win32_dl_set_search_path(const char *path);
extern HMODULE libecs_win32_dl_open(const char *filename);
extern HMODULE libecs_win32_dl_open_ext(const char *filename);
extern LPVOID libecs_win32_dl_sym(HMODULE hdl, const char *name);
extern void libecs_win32_dl_close(HMODULE hdl);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _LIBECS_WIN32_DL_COMPAT_H */
