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

#ifndef __LIBECS_WIN32_PRIVATE_H
#define __LIBECS_WIN32_PRIVATE_H

#include <windows.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct libecs_win32_mmap_hdl_list_entry_t libecs_win32_mmap_hdl_list_entry_t;
typedef struct libecs_win32_fhdl_map_entry_t libecs_win32_fhdl_map_entry_t;

struct libecs_win32_mmap_hdl_list_entry_t {
    libecs_win32_mmap_hdl_list_entry_t *next;
    libecs_win32_mmap_hdl_list_entry_t *prev;
    void *ptr;
    size_t sz;
    HANDLE hdl;
    int fd;
};

typedef struct libecs_win32_mmap_hdl_list_t {
    CRITICAL_SECTION mtx;
    libecs_win32_mmap_hdl_list_entry_t *entries;
    int num_entries; // not to use size_t is intended.
    int num_alloc;
    int num_entries_in_use;
} libecs_win32_mmap_hdl_list_t;

struct libecs_win32_fhdl_map_entry_t {
    int id;
    HANDLE hdl;
    void *first_map_ptr;
};

typedef struct libecs_win32_fhdl_map_t {
    size_t num_entries;
    size_t num_alloc;
    CRITICAL_SECTION mtx;
    libecs_win32_fhdl_map_entry_t *entries;
} libecs_win32_fhdl_map_t;

typedef struct libecs_win32_pctx_t {
    DWORD tls_idx;
    size_t page_size;
    size_t page_alloc_size;
    CRITICAL_SECTION mtx;
    char *tempdir;
    libecs_win32_fhdl_map_t fhdl_map;
    libecs_win32_mmap_hdl_list_t mhdl_list;
} libecs_win32_pctx_t;

typedef struct libecs_win32_tctx_t {
    char *err_msg;
    char *config_str;
    int last_errno;
} libecs_win32_tctx_t;

extern libecs_win32_pctx_t __libecs_win32_pctx;

extern DWORD libecs_win32_pctx_init(libecs_win32_pctx_t *ctx);
extern void libecs_win32_pctx_destroy(libecs_win32_pctx_t *ctx);

extern DWORD libecs_win32_fhdl_map_init(
        libecs_win32_fhdl_map_t *map, int initial_alloc);
extern void libecs_win32_fhdl_map_destroy(
        libecs_win32_fhdl_map_t *map);

extern libecs_win32_fhdl_map_entry_t *
libecs_win32_fhdl_map_get(libecs_win32_fhdl_map_t *map, int idx);
extern HANDLE
libecs_win32_fhdl_map_get_hdl(libecs_win32_fhdl_map_t *map, int idx);
extern void libecs_win32_fhdl_map_lock(libecs_win32_fhdl_map_t *map);
extern void libecs_win32_fhdl_map_unlock(libecs_win32_fhdl_map_t *map);
extern DWORD
libecs_win32_fhdl_map_add(libecs_win32_fhdl_map_t *map,
        libecs_win32_fhdl_map_entry_t *prototype);
extern DWORD
libecs_win32_fhdl_map_remove(libecs_win32_fhdl_map_t *map, int idx);

extern DWORD
libecs_win32_mmap_hdl_list_init(libecs_win32_mmap_hdl_list_t *self);

extern void
libecs_win32_mmap_hdl_list_destroy(libecs_win32_mmap_hdl_list_t *self);

extern libecs_win32_mmap_hdl_list_entry_t *
libecs_win32_mmap_hdl_list_add(libecs_win32_mmap_hdl_list_t *self,
        void *ptr);

extern libecs_win32_mmap_hdl_list_entry_t *
libecs_win32_mmap_hdl_list_find(libecs_win32_mmap_hdl_list_t *self,
        void *ptr);

extern void
libecs_win32_mmap_hdl_list_lock(libecs_win32_mmap_hdl_list_t *self);

extern void
libecs_win32_mmap_hdl_list_unlock(libecs_win32_mmap_hdl_list_t *self);

extern libecs_win32_tctx_t *
libecs_win32_pctx_get_tctx(libecs_win32_pctx_t *pctx);

extern int libecs_win32_translate_errno(DWORD win32_errno);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* __LIBECS_WIN32_PRIVATE_H */
