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

#include "win32_utils.h"
#include "errno.h"
#include "private.h"
#include "traits.h"

#include <windows.h>
#include <stdlib.h>
#include <strsafe.h>
#include <stdarg.h>

libecs_win32_pctx_t __libecs_win32_pctx;

DWORD libecs_win32_pctx_init(libecs_win32_pctx_t *ctx)
{
    static const DWORD keys[] = {
        STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
    };
    DWORD err;
    int i;

    {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);

        ctx->page_size = sysinfo.dwPageSize;
        ctx->page_alloc_size = sysinfo.dwAllocationGranularity;
    }

    {
        DWORD len = GetTempPath(0, NULL);
        char *tempdir;

        if (!len)
            return GetLastError();

        tempdir = libecs_win32_malloc(len);
        if (!tempdir)
            return ERROR_NOT_ENOUGH_MEMORY;
        if (!GetTempPath(len, tempdir)) {
            libecs_win32_free(tempdir);
            return GetLastError();
        }

        --len; // skip the trailing nul
        while (len > 0 && tempdir[--len] == '\\');
        if (len > 0)
            tempdir[++len] = '\0';

        ctx->tempdir = tempdir;
    }

    err = libecs_win32_fhdl_map_init(&ctx->fhdl_map, 512);
    if (err) {
        libecs_win32_free(ctx->tempdir);
        return err;
    }

    err = libecs_win32_mmap_hdl_list_init(&ctx->mhdl_list);
    if (err) {
        libecs_win32_fhdl_map_destroy(&ctx->fhdl_map);
        libecs_win32_free(ctx->tempdir);
        return err;
    }

    for (i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
        libecs_win32_fhdl_map_entry_t proto = {
            0, GetStdHandle(keys[i]), NULL
        };

        err = libecs_win32_fhdl_map_add(&ctx->fhdl_map, &proto);
        if (err) {
            libecs_win32_mmap_hdl_list_destroy(&ctx->mhdl_list);
            libecs_win32_fhdl_map_destroy(&ctx->fhdl_map);
            libecs_win32_free(ctx->tempdir);
            return err;
        }
    }

    ctx->tls_idx = TlsAlloc();
    if (ctx->tls_idx == TLS_OUT_OF_INDEXES) {
        libecs_win32_mmap_hdl_list_destroy(&ctx->mhdl_list);
        libecs_win32_fhdl_map_destroy(&ctx->fhdl_map);
        libecs_win32_free(ctx->tempdir);
        return GetLastError();
    }

    InitializeCriticalSection(&ctx->mtx);
    return ERROR_SUCCESS;
}

void libecs_win32_pctx_destroy(libecs_win32_pctx_t *ctx)
{
    DeleteCriticalSection(&ctx->mtx);
    TlsFree(ctx->tls_idx);
    libecs_win32_fhdl_map_destroy(&ctx->fhdl_map);
    libecs_win32_free(ctx->tempdir);
}


static DWORD libecs_win32_tctx_init(libecs_win32_tctx_t *tctx)
{
    tctx->err_msg = NULL;
    tctx->config_str = NULL;
    return ERROR_SUCCESS;
}

libecs_win32_tctx_t *libecs_win32_pctx_get_tctx(libecs_win32_pctx_t *pctx)
{
    libecs_win32_tctx_t *tctx =
        (libecs_win32_tctx_t *)TlsGetValue(pctx->tls_idx);

    if (!tctx) {
        tctx = libecs_win32_malloc(sizeof(libecs_win32_tctx_t));
        if (!tctx) {
            // When ran out of memory, abort the process.
            // XXX: should output some pretty error messages :-p
            abort();
        }

        if (libecs_win32_tctx_init(tctx)) {
            libecs_win32_free(tctx);
            abort();
        }

        TlsSetValue(pctx->tls_idx, tctx);
    }

    return tctx;
}

int libecs_win32_init()
{
    srand((unsigned int)GetTickCount());
    return libecs_win32_pctx_init(&__libecs_win32_pctx);
}

void libecs_win32_fini()
{
    libecs_win32_pctx_destroy(&__libecs_win32_pctx);
}

void *libecs_win32_malloc(size_t sz)
{
    return malloc(sz);
}

void *libecs_win32_alloc_array(size_t nelts, size_t elem_sz)
{
    const size_t total_size = nelts * elem_sz;

    if (total_size != (size_t)((double)elem_sz * nelts))
        return NULL;

    return libecs_win32_malloc(total_size);
}

void *libecs_win32_realloc(void *ptr, size_t sz)
{
    return realloc(ptr, sz);
}

void *libecs_win32_realloc_array(void *ptr, size_t nelts, size_t elem_sz)
{
    const size_t total_size = nelts * elem_sz;

    if (total_size != (size_t)((double)elem_sz * nelts))
        return NULL;

    return libecs_win32_realloc(ptr, total_size);
}

void libecs_win32_free(void *ptr)
{
    free(ptr);
}

char *libecs_win32_spprintf(const char *fmt, ...)
{
    va_list ap;
    char *retval;
    va_start(ap, fmt);
    retval = libecs_win32_vspprintf(fmt, ap);
    va_end(ap);
    return retval;
}

char *libecs_win32_vspprintf(const char *fmt, va_list ap)
{
    char *buf = NULL;
    size_t buf_sz = 0;
    size_t len = 0;
    HRESULT e;

    do {
        buf_sz += 16;

        buf = libecs_win32_realloc(buf, buf_sz);
        if (!buf) {
            return NULL;
        }

        // If we stick to the C spec, ap should be copy constructed
        // before passing it to StringCchVPrintf().
        // C99 specifies va_copy(), which is not provided by the
        // Microsoft's CRT. Yuck. -- moriyoshi
        e = StringCchVPrintf(buf, buf_sz, fmt, ap);
    } while (e == STRSAFE_E_INSUFFICIENT_BUFFER);

    if (e != S_OK) {
        libecs_win32_free(buf);
        return NULL;
    }

    return buf;
}

static const char *
libecs_win32_get_config_from_registry(HKEY root_key,
                                      const char *parent_key_name,
                                      const char *key_name)
{
    HKEY key;
    DWORD type, size;
    char *config_str = NULL;
    libecs_win32_tctx_t *tctx =
        libecs_win32_pctx_get_tctx(&__libecs_win32_pctx);

    if (ERROR_SUCCESS !=
            RegOpenKeyEx(root_key, parent_key_name, 0, KEY_READ, &key)) {
        return NULL;
    }

    if (ERROR_SUCCESS !=
            RegQueryValueEx(key, key_name,
                            NULL, &type, NULL, &size)) {
        goto fail;
    }

    if (type != REG_SZ) {
        goto fail;
    }

    config_str = libecs_win32_realloc(tctx->config_str, size + 1);
    if (!config_str) {
        goto fail;
    }

    if (ERROR_SUCCESS !=
            RegQueryValueEx(key, key_name,
                            NULL, NULL, config_str, &size)) {
        goto fail;
    }

    RegCloseKey(key);
    tctx->config_str = config_str;
    return config_str;

fail:
    RegCloseKey(key);
    libecs_win32_free(config_str);
    return NULL;
}

static const char *libecs_win32_get_config_from_env(const char *key)
{
    // 32767 characters maximum (NUL inclusive, officially stated)
    char buf[32767];
    char *config_str;
    libecs_win32_tctx_t *tctx =
        libecs_win32_pctx_get_tctx(&__libecs_win32_pctx);

    size_t len = GetEnvironmentVariable(key, buf, sizeof(buf));
    if (!len)
        return NULL;

    config_str = libecs_win32_malloc(len + 1);
    if (!config_str)
        return NULL;

    memcpy(config_str, buf, len);
    config_str[len] = '\0';

    tctx->config_str = config_str;
    return config_str;
}

const char *libecs_win32_get_config(const char *key)
{
    const char *config_str;

    config_str = libecs_win32_get_config_from_env(key);
    if (config_str)
        return config_str;

    config_str = libecs_win32_get_config_from_registry(
        HKEY_CURRENT_USER,
        LIBECS_WIN32_REGISTRY_KEY,
        key);
    if (config_str)
        return config_str;

    config_str = libecs_win32_get_config_from_registry(
        HKEY_LOCAL_MACHINE,
        LIBECS_WIN32_REGISTRY_KEY,
        key);
    if (config_str)
        return config_str;

    return NULL;
}

const char *libecs_win32_get_temporary_directory()
{
    return __libecs_win32_pctx.tempdir;
}

int libecs_win32_get_pid()
{
    return GetCurrentProcessId();
}

int* libecs_win32_errno()
{
    return &libecs_win32_pctx_get_tctx(&__libecs_win32_pctx)->last_errno;
}

int libecs_win32_translate_errno(DWORD win32_errno)
{
    switch (win32_errno) {
    case ERROR_SUCCESS:
        return 0;
    case ERROR_NOT_ENOUGH_MEMORY:
    case ERROR_OUTOFMEMORY:
        return _ENOMEM;
    case ERROR_ACCESS_DENIED:
    case ERROR_LOCK_VIOLATION:
        return _EACCES;
    case ERROR_PATH_NOT_FOUND:
    case ERROR_FILE_NOT_FOUND:
        return _ENOENT;
    case ERROR_ALREADY_EXISTS:
        return _EEXIST;
    case ERROR_INVALID_HANDLE:
        return _EBADF;
    case ERROR_DISK_FULL:
        return _ENOSPC;
    }

    return _EUKNWN;
}

size_t libecs_win32_get_page_size()
{
    return __libecs_win32_pctx.page_size;
}
