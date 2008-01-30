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
// part of mkstemp code derives from the FreeBSD project.
//
// Copyright (c) 1987, 1993
//	The Regents of the University of California.  All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 4. Neither the name of the University nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.
//

#include <windows.h>
#include <stdarg.h>
#include <stddef.h>
#include <ctype.h>
#include "errno.h"
#include "private.h"
#include "win32_utils.h"
#include "win32_io_compat.h"


#if defined(_WIN32_WINNT) || defined(_WIN64_WINNT)
#include <aclapi.h>
#include <sddl.h>
#include <accctrl.h>

typedef struct libecs_win32_io_ea_list_freed_entry_t
    libecs_win32_io_ea_list_freed_entry_t;

enum libecs_win32_io_ea_list_free_type_e {
    LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_NORMAL,
    LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_SID
};

struct libecs_win32_io_ea_list_freed_entry_t {
    libecs_win32_io_ea_list_freed_entry_t *next;
    enum libecs_win32_io_ea_list_free_type_e type;
    LPVOID ptr;
};

typedef struct libecs_win32_io_ea_list_t {
    size_t num_entries;
    size_t num_alloc;
    EXPLICIT_ACCESS *entries;
    libecs_win32_io_ea_list_freed_entry_t *first_free;
    libecs_win32_io_ea_list_freed_entry_t *last_free;
} libecs_win32_io_ea_list_t;

typedef struct libecs_win32_io_security_attrs_t {
    BYTE reserved[SECURITY_DESCRIPTOR_MIN_LENGTH];
    libecs_win32_io_ea_list_t list;
    PACL acl;
} libecs_win32_io_security_attrs_t;

static DWORD
libecs_win32_io_ea_list_init(libecs_win32_io_ea_list_t *list)
{
    list->num_entries = 0;
    list->num_alloc = 0;
    list->entries = NULL;
    list->first_free = list->last_free = NULL;
    return ERROR_SUCCESS;
}

static void
libecs_win32_io_ea_list_destroy(libecs_win32_io_ea_list_t *list)
{   
    libecs_win32_io_ea_list_freed_entry_t *f;
    libecs_win32_io_ea_list_freed_entry_t *nf;

    for (f = list->first_free; f; f = nf) {
        nf = f->next;
        switch (f->type) {
        case LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_NORMAL:
            libecs_win32_free(f->ptr);
            break;
        case LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_SID:
            FreeSid(f->ptr);
            break;
        }
        libecs_win32_free(f);
    }

    if (list->entries)
        libecs_win32_free(list->entries);
}

static DWORD
libecs_win32_io_ea_list_add_freed_entry(libecs_win32_io_ea_list_t *list,
        enum libecs_win32_io_ea_list_free_type_e type, LPVOID ptr)
{
    libecs_win32_io_ea_list_freed_entry_t *new_entry;

    new_entry = libecs_win32_malloc(sizeof(*new_entry));
    if (!new_entry)
        return ERROR_NOT_ENOUGH_MEMORY;

    new_entry->type = type;
    new_entry->ptr = ptr;
    new_entry->next = NULL;

    if (list->last_free) {
        list->last_free->next = new_entry;
        list->last_free = new_entry;
    } else {
        list->first_free = list->last_free = new_entry;
    }

    return ERROR_SUCCESS;
}

static EXPLICIT_ACCESS *
libecs_win32_io_ea_list_new_entry(libecs_win32_io_ea_list_t *list)
{
    if (list->num_entries >= list->num_alloc) {
        static const size_t delta = 16;
        EXPLICIT_ACCESS *new_entries;
        const size_t new_num_alloc = list->num_alloc + delta;

        if (new_num_alloc < delta)
            return NULL;

        new_entries = libecs_win32_realloc_array(list->entries,
                new_num_alloc, sizeof(EXPLICIT_ACCESS));
        if (!new_entries)
            return NULL;

        list->num_alloc = new_num_alloc;
        list->entries = new_entries;
    }

    return list->entries + (list->num_entries++);
}

static DWORD
libecs_win32_io_add_ea_entry_everyone(libecs_win32_io_ea_list_t *list, ACCESS_MASK mask)
{
    DWORD err;
    static SID_IDENTIFIER_AUTHORITY auth = SECURITY_WORLD_SID_AUTHORITY;
    PSID sid;

    if (!AllocateAndInitializeSid(&auth, 1,
            SECURITY_WORLD_RID,
            0, 0, 0, 0, 0, 0, 0,
            &sid))
        return GetLastError();

    {
        EXPLICIT_ACCESS *ea = libecs_win32_io_ea_list_new_entry(list);
        if (!ea) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto fail;
        }

        ea->grfAccessPermissions = mask;
        ea->grfAccessMode = SET_ACCESS;
        ea->grfInheritance = NO_INHERITANCE;
        ea->Trustee.pMultipleTrustee = NULL;
        ea->Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea->Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea->Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
        ea->Trustee.ptstrName = (LPTSTR) sid;

        err = libecs_win32_io_ea_list_add_freed_entry(list,
                LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_SID,
                sid);
        if (err)
            goto fail;
    }

    return ERROR_SUCCESS;

fail:
    FreeSid(sid);
    return err;
}

static DWORD
libecs_win32_io_add_ea_entry_user_and_group(libecs_win32_io_ea_list_t *list,
        ACCESS_MASK user_mask, ACCESS_MASK group_mask)
{
    HANDLE proc_acc_token = INVALID_HANDLE_VALUE;
    TOKEN_USER *user = NULL;
    TOKEN_PRIMARY_GROUP *group = NULL;
    TOKEN_GROUPS *groups = NULL;
    DWORD err = 1;

    if (!OpenProcessToken(GetCurrentProcess(),
            TOKEN_READ, &proc_acc_token))
        return GetLastError();

    {
        DWORD required;
        EXPLICIT_ACCESS *ea;
 
        // Should never success
        if (GetTokenInformation(proc_acc_token, TokenUser,
                NULL, 0, &required)) {
            err = GetLastError();
            goto out;
        }

        user = libecs_win32_malloc(required);
        if (!user) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        if (!GetTokenInformation(proc_acc_token, TokenUser,
                user, required, &required)) {
            err = GetLastError();
            goto out;
        }

        ea = libecs_win32_io_ea_list_new_entry(list);
        if (!ea) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        ea->grfAccessPermissions = user_mask;
        ea->grfAccessMode = SET_ACCESS;
        ea->grfInheritance = NO_INHERITANCE;
        ea->Trustee.pMultipleTrustee = NULL;
        ea->Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea->Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea->Trustee.TrusteeType = TRUSTEE_IS_USER;
        ea->Trustee.ptstrName = (LPTSTR) user->User.Sid;

        err = libecs_win32_io_ea_list_add_freed_entry(list,
                LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_NORMAL,
                user);
        if (err)
            goto out;
    }

    {
        DWORD required;
        EXPLICIT_ACCESS *ea;
 
        // Should never success
        if (GetTokenInformation(proc_acc_token, TokenPrimaryGroup,
                NULL, 0, &required)) {
            err = GetLastError();
            goto out;
        }

        group = libecs_win32_malloc(required);
        if (!group) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        if (!GetTokenInformation(proc_acc_token, TokenPrimaryGroup,
                group, required, &required)) {
            err = GetLastError();
            goto out;
        }

        ea = libecs_win32_io_ea_list_new_entry(list);
        if (!ea) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        ea->grfAccessPermissions = group_mask;
        ea->grfAccessMode = SET_ACCESS;
        ea->grfInheritance = NO_INHERITANCE;
        ea->Trustee.pMultipleTrustee = NULL;
        ea->Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea->Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea->Trustee.TrusteeType = TRUSTEE_IS_GROUP;
        ea->Trustee.ptstrName = (LPTSTR) group->PrimaryGroup;

        err = libecs_win32_io_ea_list_add_freed_entry(list,
                LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_NORMAL,
                group);
        if (err)
            goto out;
    }

    {
        DWORD required;
        DWORD i;

        // Should never success
        if (GetTokenInformation(proc_acc_token, TokenGroups,
                NULL, 0, &required)) {
            err = GetLastError();
            goto out;
        }

        groups = libecs_win32_malloc(required);
        if (!groups) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        if (!GetTokenInformation(proc_acc_token, TokenGroups,
                groups, required, &required)) {
            err = GetLastError();
            goto out;
        }

        for (i = 0; i < groups->GroupCount; i++) {
            EXPLICIT_ACCESS *ea;

            if (SE_GROUP_LOGON_ID !=
                    (groups->Groups[i].Attributes & SE_GROUP_LOGON_ID))
                continue;

            ea = libecs_win32_io_ea_list_new_entry(list);
            if (!ea) {
                err = ERROR_NOT_ENOUGH_MEMORY;
                goto out;
            }

            ea->grfAccessPermissions = group_mask;
            ea->grfAccessMode = SET_ACCESS;
            ea->grfInheritance = NO_INHERITANCE;
            ea->Trustee.pMultipleTrustee = NULL;
            ea->Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
            ea->Trustee.TrusteeForm = TRUSTEE_IS_SID;
            ea->Trustee.TrusteeType = TRUSTEE_IS_GROUP;
            ea->Trustee.ptstrName = (LPTSTR) groups->Groups[i].Sid;
        }

        err = libecs_win32_io_ea_list_add_freed_entry(list,
                LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_NORMAL,
                groups);
        if (err)
            goto out;
    }

    err = ERROR_SUCCESS;

out:
    if (INVALID_HANDLE_VALUE != proc_acc_token)
        CloseHandle(proc_acc_token);

    if (err) {
        if (user)
            libecs_win32_free(user);
        if (group)
            libecs_win32_free(group);
        if (groups)
            libecs_win32_free(groups);
    }

    return err;
}

static DWORD
libecs_win32_io_add_ea_entry_admin(libecs_win32_io_ea_list_t *list, ACCESS_MASK mask)
{
    static SID_IDENTIFIER_AUTHORITY auth = SECURITY_NT_AUTHORITY;
    PSID sid;
    DWORD err = 0;

    if (!AllocateAndInitializeSid(&auth, 2,
            SECURITY_BUILTIN_DOMAIN_RID, DOMAIN_ALIAS_RID_ADMINS,
            0, 0, 0, 0, 0, 0,
            &sid))
        return 1;

    {
        EXPLICIT_ACCESS *ea = libecs_win32_io_ea_list_new_entry(list);
        if (!ea) {
            err = 1;
            goto fail;
        }

        ea->grfAccessPermissions = mask;
        ea->grfAccessMode = SET_ACCESS;
        ea->grfInheritance = NO_INHERITANCE;
        ea->Trustee.pMultipleTrustee = NULL;
        ea->Trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE;
        ea->Trustee.TrusteeForm = TRUSTEE_IS_SID;
        ea->Trustee.TrusteeType = TRUSTEE_IS_GROUP;
        ea->Trustee.ptstrName = (LPTSTR) sid;

        err = libecs_win32_io_ea_list_add_freed_entry(list,
                LIBECS_WIN32_IO_EA_LIST_FREE_TYPE_SID,
                sid);
        if (err)
            goto fail;
    }

    return 0;

fail:
    FreeSid(sid);
    return err;
}

static DWORD
libecs_win32_io_build_security_attributes(PSECURITY_DESCRIPTOR *retval,
        mode_t modes)
{
    DWORD err = ERROR_SUCCESS;
    libecs_win32_io_security_attrs_t *attrs = NULL;
    int num_ea_entries = 0;

    attrs = libecs_win32_malloc(sizeof(*attrs));
    if (!attrs) {
        return ERROR_NOT_ENOUGH_MEMORY;
    }

    err = libecs_win32_io_ea_list_init(&attrs->list);
    if (err) {
        libecs_win32_free(attrs);
        return err;
    }

    attrs->acl = NULL;

    if (!InitializeSecurityDescriptor(
            (PSECURITY_DESCRIPTOR)attrs->reserved,
            SECURITY_DESCRIPTOR_REVISION)) {
        err = GetLastError();
        goto fail;
    }

    err = libecs_win32_io_add_ea_entry_admin(&attrs->list, GENERIC_ALL);
    if (err)
        goto fail;

    err = libecs_win32_io_add_ea_entry_user_and_group(&attrs->list,
            ((modes & S_IRUSR) ? GENERIC_READ: 0)
            | ((modes & S_IWUSR) ? GENERIC_WRITE: 0)
            | ((modes & S_IXUSR) ? GENERIC_EXECUTE: 0),
            ((modes & S_IRGRP) ? GENERIC_READ: 0)
            | ((modes & S_IWGRP) ? GENERIC_WRITE: 0)
            | ((modes & S_IXGRP) ? GENERIC_EXECUTE: 0));
    if (err)
        goto fail;

    err = libecs_win32_io_add_ea_entry_everyone(&attrs->list,
            ((modes & S_IROTH) ? GENERIC_READ: 0)
            | ((modes & S_IWOTH) ? GENERIC_WRITE: 0)
            | ((modes & S_IXOTH) ? GENERIC_EXECUTE: 0));
    if (err)
        goto fail;


    err = SetEntriesInAcl((ULONG)attrs->list.num_entries,
            attrs->list.entries, NULL, &attrs->acl);
    if (err)
        goto fail;

    if (!SetSecurityDescriptorDacl(
            (PSECURITY_DESCRIPTOR)attrs->reserved,
            TRUE, attrs->acl, FALSE)) {
        err = GetLastError();
        goto fail;
    }
            
    *retval = (PSECURITY_DESCRIPTOR)attrs;
    return ERROR_SUCCESS;

fail:
    libecs_win32_io_ea_list_destroy(&attrs->list);
    if (attrs->acl)
        LocalFree(attrs->acl);
    libecs_win32_free(attrs);

    return err;
}

static void libecs_win32_io_free_security_attributes(PSECURITY_DESCRIPTOR ptr)
{
    libecs_win32_io_security_attrs_t *attrs =
            (libecs_win32_io_security_attrs_t *)ptr;
    libecs_win32_io_ea_list_destroy(&attrs->list);
    LocalFree(attrs->acl);
    libecs_win32_free(ptr);
}
#endif

static DWORD libecs_win32_io_convert_path(char **retval, const char* path)
{
    size_t path_len = strlen(path);
    const char *end = path + path_len;
    char *result;
    const char *src;
    char *dst;

    // the result may be a character longer in case of absolute path
    result = libecs_win32_malloc(path_len + 2);
    if (!result)
        return ERROR_NOT_ENOUGH_MEMORY;

    src = path;
    dst = result;

    if (path_len >= 2) {
        if (path[0] == '/') {
            if (path[1] == '/') {
                // an UNC path
                result[0] = result[1] = '\\';
                src += 2;
                dst += 2;
            } else if (isalpha((int)((unsigned char *)path)[1]) &&
                    (path_len <=2 || path[2] == '/')) {
                // mingw compatibility stuff
                dst[0] = path[1];
                dst[1] = ':';
                dst[2] = '\\';
                src += 3;
                dst += 3;
            }
        }
    }

    for (; src < end; src++, dst++) {
        switch (*src) {
        case '/':
            while (++src < end && *src == '\\');
            src--;
            *dst = '\\';
            break;
        default:
            *dst = *src;
        }
    }

    *dst = '\0';
    *retval = result;

    return ERROR_SUCCESS;
}

DWORD
libecs_win32_fhdl_map_init(libecs_win32_fhdl_map_t *map,
        int initial_alloc)
{
    libecs_win32_fhdl_map_entry_t *entries = NULL;

    if (initial_alloc < 0)
        return ERROR_INVALID_DATA;

    if (initial_alloc > 0) {
        entries = libecs_win32_alloc_array(initial_alloc,
                sizeof(libecs_win32_fhdl_map_entry_t));
        if (!entries)
            return ERROR_NOT_ENOUGH_MEMORY;
    }

    map->num_alloc = (size_t)initial_alloc;
    map->num_entries = 0;
    map->entries = entries;
    InitializeCriticalSection(&map->mtx);

    return ERROR_SUCCESS;
}

void
libecs_win32_fhdl_map_destroy(libecs_win32_fhdl_map_t *map)
{
    if (map->entries)
        libecs_win32_free(map->entries);
    DeleteCriticalSection(&map->mtx);
}

libecs_win32_fhdl_map_entry_t *
libecs_win32_fhdl_map_get(libecs_win32_fhdl_map_t *map, int idx)
{
    if (idx < 0 || (size_t)idx >= map->num_entries)
        return NULL;

    return &map->entries[idx];
}

HANDLE
libecs_win32_fhdl_map_get_hdl(libecs_win32_fhdl_map_t *map, int idx)
{
    HANDLE retval = INVALID_HANDLE_VALUE;
    libecs_win32_fhdl_map_entry_t *entry;

    EnterCriticalSection(&map->mtx);

    entry = libecs_win32_fhdl_map_get(map, idx);
    if (entry)
        retval = entry->hdl;

    LeaveCriticalSection(&map->mtx);
    return retval;
}

DWORD
libecs_win32_fhdl_map_remove(libecs_win32_fhdl_map_t *map, int idx)
{
    int err = ERROR_SUCCESS;

    if (idx < 0 || (size_t)idx >= map->num_entries) {
        err = ERROR_INVALID_DATA;
        goto out;
    }

    map->entries[idx].id = -1;
    map->entries[idx].hdl = INVALID_HANDLE_VALUE;
    map->entries[idx].first_map_ptr = NULL;

    if (idx == map->num_entries - 1) {
        map->num_entries--;
    }

out:
    return err;
}

void
libecs_win32_fhdl_map_lock(libecs_win32_fhdl_map_t *map)
{
    EnterCriticalSection(&map->mtx);
}

void
libecs_win32_fhdl_map_unlock(libecs_win32_fhdl_map_t *map)
{
    LeaveCriticalSection(&map->mtx);
}

DWORD
libecs_win32_fhdl_map_add(libecs_win32_fhdl_map_t *map,
        libecs_win32_fhdl_map_entry_t *prototype)
{
    DWORD err = ERROR_SUCCESS;

    EnterCriticalSection(&map->mtx);

    if (map->num_entries >= map->num_alloc) {
        libecs_win32_fhdl_map_entry_t *new_entries = NULL;
        static const size_t growth = 256;
        const size_t new_num_alloc = map->num_alloc + growth;

        if (new_num_alloc < growth) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        new_entries = libecs_win32_realloc_array(
                map->entries, new_num_alloc,
                sizeof(libecs_win32_fhdl_map_entry_t));
        if (!new_entries) {
            err = ERROR_NOT_ENOUGH_MEMORY;
            goto out;
        }

        map->entries = new_entries;
    }

    prototype->id = (int)map->num_entries;
    map->entries[map->num_entries++] = *prototype;

out:
    LeaveCriticalSection(&map->mtx);
    return err;
}

int libecs_win32_io_open(const char *pathname, int flags, ...)
{
    char *real_pathname;
    va_list ap;
    DWORD err = ERROR_SUCCESS;
    mode_t modes = 0666;
    HANDLE hdl;
    DWORD desired_access = 0;
    DWORD creation_disposition = 0;
    DWORD attr = FILE_ATTRIBUTE_ARCHIVE;
    SECURITY_ATTRIBUTES sec_attr = { sizeof(SECURITY_ATTRIBUTES), 0, TRUE };

    err = libecs_win32_io_convert_path(&real_pathname, pathname);
    if (err) {
        __libecs_errno = libecs_win32_translate_errno(err);
        return -1;
    }

    va_start(ap, flags);
    if (flags & O_CREAT)
        modes = va_arg(ap, mode_t);
    va_end(ap);

    if (flags & O_RDONLY)
        desired_access |= FILE_SHARE_READ;
    if (flags & O_WRONLY)
        desired_access |= FILE_SHARE_WRITE;
    if (flags & O_CREAT) {
        if (flags & O_EXCL)
            creation_disposition = CREATE_NEW;
        else {
            if (flags & O_TRUNC)
                creation_disposition = CREATE_ALWAYS;
            else
                creation_disposition = OPEN_ALWAYS;
        }

#if defined(_WIN32_WINNT) || defined(_WIN64_WINNT)
        err = libecs_win32_io_build_security_attributes(
                &sec_attr.lpSecurityDescriptor, modes);
        if (err)
            goto out;
#else
        sec_attr.lpSecurityDescriptor = NULL;
#endif
    } else {
        if (flags & O_TRUNC)
            creation_disposition = TRUNCATE_EXISTING;
        else
            creation_disposition = OPEN_EXISTING;

        sec_attr.lpSecurityDescriptor = NULL;
    }

    if (0 == (modes & 0222))
        attr |= FILE_ATTRIBUTE_READONLY;
    if (flags & O_DIRECT)
        attr |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS;
    else
        attr |= FILE_FLAG_SEQUENTIAL_SCAN;
    if (flags & O_SYNC)
        attr |= FILE_FLAG_WRITE_THROUGH;

    hdl = CreateFile(real_pathname, desired_access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            &sec_attr, creation_disposition, attr,
            NULL);

    if (INVALID_HANDLE_VALUE == hdl)
        err = GetLastError();

out:
    if (real_pathname)
        libecs_win32_free(real_pathname);

#if defined(_WIN32_WINNT) || defined(_WIN64_WINNT)
    if (sec_attr.lpSecurityDescriptor) {
        libecs_win32_io_free_security_attributes(
            sec_attr.lpSecurityDescriptor);
    }
#endif

    if (err) {
        __libecs_errno = libecs_win32_translate_errno(err);
        return -1;
    }

    {
        libecs_win32_fhdl_map_entry_t proto = { 0, hdl, NULL };

        err = libecs_win32_fhdl_map_add(
                &__libecs_win32_pctx.fhdl_map,
                &proto);
        if (err) {
            CloseHandle(hdl);
            __libecs_errno = libecs_win32_translate_errno(err);
            return -1;
        }

        return proto.id;
    }
}

int libecs_win32_io_creat(const char *pathname, mode_t mode)
{
    return libecs_win32_io_open(pathname, O_CREAT | O_WRONLY | O_TRUNC, mode);
}
 
int libecs_win32_io_close(int fd)
{
    libecs_win32_fhdl_map_entry_t *hdl_ent;

    libecs_win32_fhdl_map_lock(&__libecs_win32_pctx.fhdl_map);

    hdl_ent = libecs_win32_fhdl_map_get(
            &__libecs_win32_pctx.fhdl_map, fd);
    if (!hdl_ent) {
        libecs_win32_fhdl_map_unlock(&__libecs_win32_pctx.fhdl_map);
        __libecs_errno = _EBADF;        
        return -1;
    }

    // Destroy all the associated mappings.
    {
        libecs_win32_mmap_hdl_list_entry_t *entry, *next;
        int num_entries_gone = 0;

        libecs_win32_mmap_hdl_list_lock(
                &__libecs_win32_pctx.mhdl_list);
        entry = libecs_win32_mmap_hdl_list_find(
                &__libecs_win32_pctx.mhdl_list,
                hdl_ent->first_map_ptr);
        for (; entry; entry = next) {
            next = entry->next;
            UnmapViewOfFile(entry->ptr);
            CloseHandle(entry->hdl);

            entry->hdl = INVALID_HANDLE_VALUE;
            entry->prev = entry->next = NULL;
            entry->fd = -1;

            num_entries_gone++;
        }

        __libecs_win32_pctx.mhdl_list.num_entries_in_use -= num_entries_gone;

        if (!__libecs_win32_pctx.mhdl_list.num_entries_in_use) {
            __libecs_win32_pctx.mhdl_list.num_entries = 0;
        }

        libecs_win32_mmap_hdl_list_unlock(
                &__libecs_win32_pctx.mhdl_list);
    }

    if (!CloseHandle(hdl_ent->hdl)) {
        libecs_win32_fhdl_map_unlock(&__libecs_win32_pctx.fhdl_map);
        __libecs_errno = libecs_win32_translate_errno(GetLastError());
        return -1;
    }

    libecs_win32_fhdl_map_remove(&__libecs_win32_pctx.fhdl_map, fd);
    libecs_win32_fhdl_map_unlock(&__libecs_win32_pctx.fhdl_map);


    return 0;
}

int libecs_win32_io_read(int fd, void *buf, size_t size)
{
    DWORD retval;
    HANDLE hdl;

    if (size > INT_MAX) {
        __libecs_errno = _EINVAL;
        return -1;
    }

    hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
    if (INVALID_HANDLE_VALUE == hdl) {
        __libecs_errno = _EBADF;        
        return -1;
    }

    if (!ReadFile(hdl, buf, (DWORD)size, &retval, NULL)) {
        __libecs_errno = libecs_win32_translate_errno(GetLastError());
        return -1;
    }

    return retval;
}

int libecs_win32_io_write(int fd, const void *buf, size_t size)
{
    DWORD retval;
    HANDLE hdl;

    if (size > INT_MAX) {
        __libecs_errno = _EINVAL;
        return -1;
    }

    hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
    if (INVALID_HANDLE_VALUE == hdl) {
        __libecs_errno = _EBADF;        
        return -1;
    }

    if (!WriteFile(hdl, buf, (DWORD)size, &retval, NULL)) {
        __libecs_errno = libecs_win32_translate_errno(GetLastError());
        return -1;
    }

    return retval;
}

static DWORD libecs_win32_io_get_whence_code(int w)
{
    switch (w) {
    case SEEK_SET:
        return FILE_BEGIN;
    case SEEK_CUR:
        return FILE_CURRENT;
    case SEEK_END:
        return FILE_END;
    }

    return -1;
}

static DWORD
libecs_win32_io_calculate_file_offset(HANDLE hdl,
        OVERLAPPED* retval, off_t offset, int whence)
{
    switch (whence) {
    case SEEK_SET:
        retval->Offset = (DWORD)(offset & (off_t)((DWORD)~0));
        retval->OffsetHigh = (DWORD)(offset >> (sizeof(DWORD) * 8));
        break;

    case SEEK_CUR:
        {
#if _WIN32_WINNT >= 0x500 || _WIN64_WINNT >= 0x500
            LARGE_INTEGER l;
            static const LARGE_INTEGER li_zero = { 0, 0 };
            if (!SetFilePointerEx(hdl, li_zero,
                    &l, FILE_CURRENT)) {
                return GetLastError();
            }

            l.QuadPart += offset;
            retval->Offset = l.LowPart;
            retval->OffsetHigh = l.HighPart;
#else
            LONG l;
            if (!SetFilePointer(hdl, 0,
                    &l, FILE_CURRENT)) {
                return GetLastError();
            } 

            if (offset > 0) {                  
                retval->Offset = (DWORD)(offset & (off_t)((DWORD)~0)) + l;
                retval->OffsetHigh = (DWORD)(offset >> (sizeof(DWORD) * 8))
                        + (retval->Offset < (DWORD)l ? 1: 0);
            } else {
                retval->Offset = (DWORD)l - (DWORD)(offset & (off_t)((DWORD)~0));
                retval->OffsetHigh = (DWORD)(offset >> (sizeof(DWORD) * 8))
                        - (retval->Offset > (DWORD)l ? 1: 0);
            }
#endif
        }
        break;
    case SEEK_END:
        {
#if _WIN32_WINNT >= 0x500 || _WIN64_WINNT >= 0x500
            LARGE_INTEGER l;

            if (!GetFileSizeEx(hdl, &l)) {
                return GetLastError();
            }

            l.QuadPart += offset;
            retval->Offset = l.LowPart;
            retval->OffsetHigh = l.HighPart;
#else
            LONG l;
            if (!GetFileSize(hdl, &l)) {
                return GetLastError();
            }                   

            if (offset > 0) {
                retval->Offset = (DWORD)(offset & (off_t)((DWORD)~0)) + l;
                retval->OffsetHigh = (DWORD)(offset >> (sizeof(DWORD) * 8))
                        + (retval->Offset < (DWORD)l ? 1: 0);
            } else {
                retval->Offset = (DWORD)l - (DWORD)(offset & (off_t)((DWORD)~0));
                retval->OffsetHigh = (DWORD)(offset >> (sizeof(DWORD) * 8))
                        - (retval->Offset > (DWORD)l ? 1: 0);
            }
#endif
        }
        break;
    }

    return ERROR_SUCCESS;
}

int libecs_win32_io_cntl(int fd, int cmd, ...)
{
    va_list ap;
    HANDLE hdl;

    hdl = libecs_win32_fhdl_map_get(&__libecs_win32_pctx.fhdl_map, fd);
    if (!hdl) {
        __libecs_errno = _EBADF;
        return -1;
    }

    switch (cmd) {
    case F_DUPFD:
        {
            DWORD err;
            HANDLE proc_hdl = GetCurrentProcess();
            libecs_win32_fhdl_map_entry_t proto = { 0, INVALID_HANDLE_VALUE, NULL };

            if (!DuplicateHandle(
                    proc_hdl, hdl,
                    proc_hdl, &proto.hdl, 0,
                    TRUE, DUPLICATE_SAME_ACCESS)) {
                __libecs_errno = libecs_win32_translate_errno(GetLastError());        
                return -1;
            }

            err = libecs_win32_fhdl_map_add(
                    &__libecs_win32_pctx.fhdl_map,
                    &proto);
            if (err) {
                CloseHandle(proto.hdl);
                __libecs_errno = libecs_win32_translate_errno(err);
                return -1;
            }

            return proto.id;
        }
        break;
    case F_GETFD:
        __libecs_errno = _EINVAL;
        return -1;
    case F_SETFD:
        __libecs_errno = _EINVAL;
        return -1;
    case F_GETLK:
        __libecs_errno = _EINVAL;
        return -1;
    case F_SETLK:
    case F_SETLKW:
        {
            DWORD err;
            struct flock *fld;
            OVERLAPPED offset;

            va_start(ap, cmd);
            fld = va_arg(ap, struct flock *);
            va_end(ap);

            err = libecs_win32_io_calculate_file_offset(hdl,
                    &offset, fld->l_start, fld->l_whence);
            if (err) {
                __libecs_errno = libecs_win32_translate_errno(err);
                return -1;
            }

            if (fld->l_type == F_UNLCK) {
                if (!UnlockFileEx(hdl, 0,
                        (DWORD)(fld->l_len & (off_t)((DWORD)~0)),
                        (DWORD)(fld->l_len >> (sizeof(DWORD) * 8)),
                        &offset)) {
                    __libecs_errno = libecs_win32_translate_errno(GetLastError());
                    return -1;
                }
            } else {
                if (!LockFileEx(hdl,
                        ((fld->l_type == F_WRLCK) ?
                            LOCKFILE_EXCLUSIVE_LOCK: 0)
                        | (cmd == F_SETLK ?
                            LOCKFILE_FAIL_IMMEDIATELY: 0),
                        0,
                        (DWORD)(fld->l_len & (off_t)((DWORD)~0)),
                        (DWORD)(fld->l_len >> (sizeof(DWORD) * 8)),
                        &offset)) {
                    __libecs_errno = libecs_win32_translate_errno(GetLastError());
                    return -1;
                }
            }
        }
        break;
    }

    return 0;
}

off_t libecs_win32_io_seek(int fd, off_t offset, int whence)
{
    HANDLE hdl;

    hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
    if (INVALID_HANDLE_VALUE == hdl) {
        __libecs_errno = _EBADF;        
        return -1;
    }

    {
#if _WIN32_WINNT >= 0x500 || _WIN64_WINNT >= 0x500
        LARGE_INTEGER _offset, new_offset;
        _offset.QuadPart = offset;

        if (!SetFilePointerEx(hdl, _offset, &new_offset,
                libecs_win32_io_get_whence_code(whence))) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        return _offset.QuadPart;
#else
        LONG new_offset;
        if (!SetFilePointer(hdl, (LONG)offset, &new_offset,
                libecs_win32_io_get_whence_code(whence))) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        return new_offset;
#endif
    }
}

#include <stdlib.h>

int libecs_win32_io_mkstemps(char *tpl, size_t slen)
{
    static const unsigned char padchar[] =
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	char *start, *trv, *suffp;
	char *pad;
    int fd = -1;

    trv = tpl + strlen(tpl);
	trv -= slen;
	suffp = trv;
	--trv;
	if (trv < tpl) {
		__libecs_errno = _EINVAL;
        return -1;
	}

	/* Fill space with random characters */
	while (trv >= tpl && *trv == 'X') {
		size_t r = rand() % (sizeof(padchar) - 1);
		*trv-- = padchar[r];
	}
	start = trv + 1;

	for (;;) {
		fd = libecs_win32_io_open(tpl, O_CREAT|O_EXCL|O_RDWR, 0600);
        if (fd >= 0)
            break;
        if (__libecs_errno != _EEXIST) {
			return -1;
        }

		/* If we have a collision, cycle through the space of filenames */
		for (trv = start;;) {
			if (*trv == '\0' || trv == suffp)
				return 0;
			pad = strchr(padchar, *trv);
			if (pad == NULL || *++pad == '\0')
				*trv++ = padchar[0];
			else {
				*trv++ = *pad;
				break;
			}
		}
	}

    return fd;
}

int libecs_win32_io_truncate(int fd, off_t length)
{
    HANDLE hdl;

    hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
    if (INVALID_HANDLE_VALUE == hdl) {
        __libecs_errno = _EBADF;        
        return -1;
    }

    {
#if _WIN32_WINNT >= 0x500 || _WIN64_WINNT >= 0x500
        LARGE_INTEGER _length, current_offset;
        static LARGE_INTEGER li_zero = { 0, 0 };

        if (!SetFilePointerEx(hdl, li_zero,
                &current_offset, FILE_CURRENT)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        _length.QuadPart = length;

        if (!SetFilePointerEx(hdl, _length, NULL,
                FILE_BEGIN)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        if (!SetEndOfFile(hdl)) {
            DWORD last_error = GetLastError();
            SetFilePointerEx(hdl, current_offset,
                    NULL, FILE_BEGIN);
            __libecs_errno = libecs_win32_translate_errno(last_error);
            return -1;
        }

        if (!SetFilePointerEx(hdl, current_offset,
                NULL, FILE_BEGIN)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }
#else
        LONG _length = (LONG)length, current_offset;
        if (!SetFilePointer(hdl, 0, &current_offset,
                FILE_CURRENT)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        if (!SetFilePointer(hdl, _length, NULL,
                FILE_BEGIN)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }

        if (!SetEndOfFile(hdl)) {
            DWORD last_error = GetLastError();
            SetFilePointer(hdl, current_offset,
                    NULL, FILE_BEGIN);
            __libecs_errno = libecs_win32_translate_errno(last_error);
            return -1;
        }

        if (!SetFilePointer(hdl, current_offset,
                NULL, FILE_BEGIN)) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError());
            return -1;
        }
#endif
    }

    return 0;
}

static void convert_to_unix_ts(FILETIME *ft, struct timespec *ts)
{
    struct timespec _ts;
    /* the magic number is the period in 10^-7 seconds
     * between 1601/1/1 - 1970/1/1 */
    DWORD h = ft->dwHighDateTime - 27111902;
    DWORD l = ft->dwLowDateTime - 3577643008;
    DWORD td, tr, ud, ur;

    if (h > ft->dwHighDateTime) {
        ts->tv_sec = ts->tv_nsec = 0;
        return;
    }

    if (l > ft->dwLowDateTime) {
        if (h == 0) {
            ts->tv_sec = ts->tv_nsec = 0;
            return;
        }
        h--;
    }

    /* split a large div operation into several steps to
     * avoid integer overflow :( */
    _ts.tv_sec = h * 429;
    td = h / 256, tr = h % 256;
    ud = td / 492, ur = td % 492;
    _ts.tv_sec += td * 127 + ud * 80;
    _ts.tv_nsec = ud * 865792 + ur * 1627776 + tr * 4967296;
    if (_ts.tv_nsec < 0) {
        // carry out
        _ts.tv_sec += 215;
        _ts.tv_nsec = (_ts.tv_nsec - (1 << (sizeof(_ts.tv_nsec) * 8 - 1))) - 2516352;
    }
    _ts.tv_nsec += l;
    if (_ts.tv_nsec < 0) {
        // carry out
        _ts.tv_sec += 215;
        _ts.tv_nsec = (_ts.tv_nsec - (1 << (sizeof(_ts.tv_nsec) * 8 - 1))) - 2516352;
    }

    if (_ts.tv_sec < 0) {
        ts->tv_sec = ts->tv_nsec = 0;
        return;
    }

    _ts.tv_sec += _ts.tv_nsec / 10000000;
    _ts.tv_nsec %= 10000000;
    _ts.tv_nsec *= 100;

    *ts = _ts;
}

extern int libecs_win32_io_stat(int fd, struct stat *sb)
{
    HANDLE hdl;
    BY_HANDLE_FILE_INFORMATION fi;

    hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
    if (INVALID_HANDLE_VALUE == hdl) {
        __libecs_errno = _EBADF;        
        return -1;
    }

    GetFileInformationByHandle(hdl, &fi);

    sb->st_dev = fi.dwVolumeSerialNumber;
    sb->st_ino = (ino_t)fi.nFileIndexLow | ((ino_t)fi.nFileIndexHigh << 32);
    sb->st_size = (off_t)fi.nFileSizeLow | ((off_t)fi.nFileSizeHigh << 32);

    convert_to_unix_ts(
            &fi.ftLastAccessTime,
            &sb->st_atimespec);
    sb->st_atime = sb->st_atimespec.tv_sec;
    convert_to_unix_ts(
            &fi.ftCreationTime,
            &sb->st_ctimespec);
    sb->st_ctime = sb->st_ctimespec.tv_sec;
    convert_to_unix_ts(
            &fi.ftLastWriteTime,
            &sb->st_mtimespec);
    sb->st_mtime = sb->st_mtimespec.tv_sec;

    return 0;
}


int libecs_win32_io_unlink(const char *path)
{
    char *real_path;
    DWORD err = libecs_win32_io_convert_path(&real_path, path);

    if (!DeleteFile(real_path)) {
        __libecs_errno = libecs_win32_translate_errno(GetLastError());
        libecs_win32_free(real_path);
        return -1;
    }

    libecs_win32_free(real_path);
    return 0;
}
