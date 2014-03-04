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
//
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
//

#ifndef __LIBECS_WIN32_UTILS_H
#define __LIBECS_WIN32_UTILS_H

#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// Utility functions
extern int libecs_win32_init();
extern void libecs_win32_fini();

extern void *libecs_win32_malloc(size_t sz);
extern void *libecs_win32_alloc_array(size_t nelts, size_t elem_sz);
extern void *libecs_win32_realloc(void *ptr, size_t sz);
extern void *libecs_win32_realloc_array(void *ptr, size_t nelts, size_t elem_sz);
extern void libecs_win32_free(void *ptr);
extern char *libecs_win32_spprintf(const char *fmt, ...);
extern char *libecs_win32_vspprintf(const char *fmt, va_list ap);
extern const char *libecs_win32_get_temporary_directory();
extern const char *libecs_win32_get_config(const char *key);
extern int libecs_win32_get_pid();
extern size_t libecs_win32_get_page_size();

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* __LIBECS_WIN32_UTILS_H */
