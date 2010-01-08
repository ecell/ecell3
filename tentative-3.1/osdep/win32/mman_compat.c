//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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

#include <assert.h>
#include "errno.h"
#include "private.h"
#include "win32_io_compat.h"
#include "win32_mman_compat.h"
#include "win32_utils.h"

DWORD libecs_win32_mmap_hdl_list_init(libecs_win32_mmap_hdl_list_t *self)
{
    InitializeCriticalSection(&self->mtx);
    self->num_alloc = self->num_entries = self->num_entries_in_use = 0;
    self->entries = NULL;

    return ERROR_SUCCESS;
}

void libecs_win32_mmap_hdl_list_destroy(libecs_win32_mmap_hdl_list_t *self)
{
    EnterCriticalSection(&self->mtx);

    if (self->entries)
        libecs_win32_free(self->entries);

    self->entries = NULL;

    LeaveCriticalSection(&self->mtx);
    DeleteCriticalSection(&self->mtx);
}

libecs_win32_mmap_hdl_list_entry_t *
libecs_win32_mmap_hdl_list_add(libecs_win32_mmap_hdl_list_t *self, void *ptr)
{
    int e = self->num_entries, s = 0, l = 0;
    libecs_win32_mmap_hdl_list_entry_t *entry;

    if (s == e) {
        const int initial_alloc = 16;
        libecs_win32_mmap_hdl_list_entry_t *new_entries;

        new_entries = libecs_win32_alloc_array(
                initial_alloc, sizeof(*entry));
        if (!new_entries)
            return NULL;

        self->entries = new_entries;
        self->num_alloc = initial_alloc;
        self->num_entries = 1;
    } else {
        void *p = 0;

        while (s <= e) {
            l = s + (e - s) / 2;
            p = self->entries[l].ptr;

            if (p == ptr) {
                assert(self->entries[l].hdl == INVALID_HANDLE_VALUE);
                break;
            } else if (p < ptr) {
                s = l + 1;
            } else {
                e = l - 1;
            }
        }

        if (p == ptr) {
            ; // do nothing
        } else if (l + 1 < self->num_entries &&
                INVALID_HANDLE_VALUE == self->entries[l + 1].hdl) {
            l++;
        } else if (l > 0 &&
                INVALID_HANDLE_VALUE == self->entries[l - 1].hdl) {
            l--;
        } else {
            libecs_win32_mmap_hdl_list_entry_t *new_entries = self->entries;
            libecs_win32_mmap_hdl_list_entry_t *old_entries = self->entries;

            if (self->num_entries >= self->num_alloc) {
                const int growth = 16;
                int new_num_alloc = self->num_alloc + growth;

                if (new_num_alloc < growth)
                    return NULL;

                new_entries = libecs_win32_realloc_array(self->entries,
                        new_num_alloc, sizeof(*entry));
                if (!new_entries)
                    return NULL;

                self->entries = new_entries;
                self->num_alloc = new_num_alloc;
            }
            if (l < self->num_entries) {
                if (new_entries == old_entries) {
                    libecs_win32_mmap_hdl_list_entry_t *old_sep_entry =
                            &old_entries[l];

                    int i = self->num_entries;
                    while (--i >= l) {
                        libecs_win32_mmap_hdl_list_entry_t *entry = &self->entries[i];
                        if (entry->next >= old_sep_entry)
                            entry->next++;
                        if (entry->prev >= old_sep_entry)
                            entry->prev++;
                        *(entry + 1) = *entry;
                    }

                    while (--i >= 0) {
                        libecs_win32_mmap_hdl_list_entry_t *entry = &self->entries[i];
                        if (entry->next >= old_sep_entry)
                            entry->next++;
                        if (entry->prev >= old_sep_entry)
                            entry->prev++;
                    }
                } else {
                    libecs_win32_mmap_hdl_list_entry_t *old_sep_entry =
                            &old_entries[l];
                    int i = self->num_entries;
                    while (--i >= l) {
                        libecs_win32_mmap_hdl_list_entry_t *entry = &self->entries[i];
                        if (entry->next) {
                            entry->next = new_entries + (entry->next - old_entries)
                                    + (entry->next >= old_sep_entry ? 1: 0);
                        }
                        if (entry->prev) {
                            entry->prev = new_entries + (entry->prev - old_entries)
                                    + (entry->prev >= old_sep_entry ? 1: 0);
                        }
                        *(entry + 1) = *entry;
                    }

                    while (--i >= 0) {
                        libecs_win32_mmap_hdl_list_entry_t *entry = &self->entries[i];
                        if (entry->next) {
                            entry->next = new_entries + (entry->next - old_entries)
                                    + (entry->next >= old_sep_entry ? 1: 0);
                        }
                        if (entry->prev) {
                            entry->prev = new_entries + (entry->prev - old_entries)
                                    + (entry->prev >= old_sep_entry ? 1: 0);
                        }
                    }
                }
            } else if (new_entries != old_entries) {
                int i = self->num_entries;
                while (--i >= 0) {
                    libecs_win32_mmap_hdl_list_entry_t *entry =
                            &self->entries[i];
                    if (entry->next)
                        entry->next = new_entries + (entry->next - old_entries);
                    if (entry->prev)
                        entry->prev = new_entries + (entry->prev - old_entries);
                }
            }

            self->num_entries++;
        }
    }

    entry = &self->entries[l];

    entry->hdl = INVALID_HANDLE_VALUE;
    entry->ptr = ptr;
    entry->sz  = 0;
    entry->fd  = -1;
    entry->next = entry->prev = NULL;

    return entry;
}

libecs_win32_mmap_hdl_list_entry_t *
libecs_win32_mmap_hdl_list_find(libecs_win32_mmap_hdl_list_t *self,
        void *ptr)
{
    int e = self->num_entries, s = 0, l = 0;

    if (e == 0)
        return NULL;

    while (s <= e) { 
        void *p;
        libecs_win32_mmap_hdl_list_entry_t *entry = &self->entries[l];
        l = s + (e - s) / 2;
        p = entry->ptr;

        if (p == ptr) {
            if (INVALID_HANDLE_VALUE != entry->hdl)
                return entry; 
            break;
        } else if (p < ptr) {
            s = l + 1;
        } else {
            e = l - 1;
        }
    }

    return NULL;
}

void
libecs_win32_mmap_hdl_list_lock(libecs_win32_mmap_hdl_list_t *self)
{
    EnterCriticalSection(&self->mtx);
}

void
libecs_win32_mmap_hdl_list_unlock(libecs_win32_mmap_hdl_list_t *self)
{
    LeaveCriticalSection(&self->mtx);
}

void *libecs_win32_mman_map(void *start, size_t size, int protection,
        int flags, int fd, off_t offset)
{
    DWORD prot_flag, acc_flag;

    if (offset < 0) {
        __libecs_errno = _EINVAL;
        return MAP_FAILED;
    }

    protection &= PROT_MASK;

    if (protection & PROT_READ) {
        if (protection & PROT_WRITE) {
            if (protection & PROT_EXEC) {
                if (flags & MAP_PRIVATE) {
                    acc_flag = FILE_MAP_READ | FILE_MAP_WRITE | FILE_MAP_EXECUTE | FILE_MAP_COPY;
                    prot_flag = PAGE_EXECUTE_WRITECOPY;
                } else {
                    acc_flag = FILE_MAP_READ | FILE_MAP_WRITE | FILE_MAP_EXECUTE;
                    prot_flag = PAGE_EXECUTE_READWRITE;
                }
            } else {
                if (flags & MAP_PRIVATE) {
                    acc_flag = FILE_MAP_READ | FILE_MAP_WRITE | FILE_MAP_COPY;
                    prot_flag = PAGE_WRITECOPY;
                } else {
                    acc_flag = FILE_MAP_READ | FILE_MAP_WRITE;
                    prot_flag = PAGE_READWRITE;
                }
            }
        } else {
            if (protection & PROT_EXEC) {
                acc_flag = FILE_MAP_READ | FILE_MAP_EXECUTE;
                prot_flag = PAGE_EXECUTE_READ;
            } else {
                acc_flag = FILE_MAP_READ;
                prot_flag = PAGE_READONLY;
            }
        }
    } else {
        if (protection & PROT_WRITE) {
            if (protection & PROT_EXEC) {
                errno = _EINVAL;
                return NULL;
            } else {
                if (flags & MAP_PRIVATE) {
                    acc_flag = FILE_MAP_WRITE | FILE_MAP_COPY;
                    prot_flag = PAGE_WRITECOPY;
                } else {
                    acc_flag = FILE_MAP_WRITE;
                    prot_flag = PAGE_READWRITE;
                }
            }
        } else {
            acc_flag = 0;
            prot_flag = PAGE_NOACCESS;
        }
    }

    if (flags & MAP_ANONYMOUS) {
        LPVOID retval = VirtualAlloc(start, size, MEM_COMMIT | MEM_RESERVE, prot_flag);

        if (!retval) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError()); 
            return MAP_FAILED;
        }

        return retval;
    } else {
        HANDLE hdl, mhdl;
        LPVOID region;
        libecs_win32_mmap_hdl_list_entry_t *entry;
        libecs_win32_fhdl_map_entry_t *hdl_ent;
        off_t map_offset = offset & ~((off_t)__libecs_win32_pctx.page_alloc_size-1);
        const size_t gap = (size_t)(offset - map_offset);
        size += gap;

        hdl = libecs_win32_fhdl_map_get_hdl(&__libecs_win32_pctx.fhdl_map, fd);
        if (INVALID_HANDLE_VALUE == hdl) {
            libecs_win32_fhdl_map_unlock(&__libecs_win32_pctx.fhdl_map);
            __libecs_errno = _EBADF;
            return MAP_FAILED;
        }

        libecs_win32_mmap_hdl_list_lock(&__libecs_win32_pctx.mhdl_list);

        mhdl = CreateFileMapping(hdl, NULL, prot_flag, 0, 0, NULL);
        if (!mhdl) {
            __libecs_errno = libecs_win32_translate_errno(GetLastError()); 
            libecs_win32_mmap_hdl_list_unlock(&__libecs_win32_pctx.mhdl_list);
            return MAP_FAILED;
        }

        region = MapViewOfFileEx(mhdl, acc_flag,
                (DWORD)(map_offset >> 32),
                (DWORD)(map_offset & (off_t)((DWORD)~0)),
                size, start);
        if (!region) {
            CloseHandle(mhdl);
            __libecs_errno = libecs_win32_translate_errno(GetLastError()); 
            libecs_win32_mmap_hdl_list_unlock(&__libecs_win32_pctx.mhdl_list);
            return MAP_FAILED;
        }

        entry = libecs_win32_mmap_hdl_list_add(
                &__libecs_win32_pctx.mhdl_list, region);                
        if (!entry) {
            UnmapViewOfFile(region);
            CloseHandle(mhdl);
            __libecs_errno = _ENOMEM;
            libecs_win32_mmap_hdl_list_unlock(&__libecs_win32_pctx.mhdl_list);
            return MAP_FAILED;
        }

        libecs_win32_fhdl_map_lock(&__libecs_win32_pctx.fhdl_map);
        {
            libecs_win32_mmap_hdl_list_entry_t *first = NULL;

            hdl_ent = libecs_win32_fhdl_map_get(
                    &__libecs_win32_pctx.fhdl_map, fd);
            if (!hdl_ent) {
                entry->hdl = NULL;
                UnmapViewOfFile(region);
                CloseHandle(mhdl);
                libecs_win32_mmap_hdl_list_unlock(
                        &__libecs_win32_pctx.mhdl_list);
                libecs_win32_fhdl_map_unlock(
                        &__libecs_win32_pctx.fhdl_map);
                __libecs_errno = _EBADF;
                return MAP_FAILED;
            }

            if (hdl_ent->first_map_ptr) {
                first = libecs_win32_mmap_hdl_list_find(
                        &__libecs_win32_pctx.mhdl_list,
                        hdl_ent->first_map_ptr);
                assert(first != NULL);
            }

            entry->fd = fd;
            entry->hdl = mhdl;
            entry->ptr = region;
            entry->sz = size;

            // add to the linked list
            entry->next = first;
            if (entry->next) {
                entry->next->prev = entry;
            }
            hdl_ent->first_map_ptr = entry->ptr;

            __libecs_win32_pctx.mhdl_list.num_entries_in_use++;

            libecs_win32_mmap_hdl_list_unlock(
                    &__libecs_win32_pctx.mhdl_list);
        }
        libecs_win32_fhdl_map_unlock(&__libecs_win32_pctx.fhdl_map);

        return ((BYTE *)region) + gap;
    }

    // not reached
}

int libecs_win32_mman_unmap(void *start, size_t length)
{
    MEMORY_BASIC_INFORMATION mbi;

    start = (void *)(((UINT_PTR)start) &
            ~(__libecs_win32_pctx.page_alloc_size - 1));

    VirtualQuery(start, &mbi, sizeof(mbi));

    if (length > mbi.RegionSize) {
        __libecs_errno = _EINVAL;
        return -1;
    }

    if (mbi.Type & MEM_MAPPED) {
        libecs_win32_mmap_hdl_list_entry_t *entry;
        libecs_win32_fhdl_map_entry_t *fhdl_ent;
        HANDLE mhdl;

        libecs_win32_mmap_hdl_list_lock(
                &__libecs_win32_pctx.mhdl_list);

        entry = libecs_win32_mmap_hdl_list_find(
                &__libecs_win32_pctx.mhdl_list, start);
        if (!entry) {
            libecs_win32_mmap_hdl_list_unlock(
                    &__libecs_win32_pctx.mhdl_list);
            __libecs_errno = _EINVAL;
            return -1;
        }

        if (INVALID_HANDLE_VALUE == entry->hdl) {
            libecs_win32_mmap_hdl_list_unlock(
                    &__libecs_win32_pctx.mhdl_list);
            __libecs_errno = _EINVAL;
            return -1;
        }

        mhdl = entry->hdl;

        entry->hdl = INVALID_HANDLE_VALUE;
        if (!entry->prev) {
            libecs_win32_fhdl_map_lock(
                    &__libecs_win32_pctx.fhdl_map);
            fhdl_ent = libecs_win32_fhdl_map_get(
                    &__libecs_win32_pctx.fhdl_map, entry->fd);
            assert(fhdl_ent);
            fhdl_ent->first_map_ptr = entry->next ? entry->next->ptr: NULL;
            libecs_win32_fhdl_map_unlock(
                    &__libecs_win32_pctx.fhdl_map);
        } else {
            entry->prev->next = entry->next;
        }

        if (entry->next) {
            entry->next->prev = entry->prev;
        }

        entry->next = entry->prev = NULL;
        entry->fd = -1;

        __libecs_win32_pctx.mhdl_list.num_entries_in_use--;

        if (entry - __libecs_win32_pctx.mhdl_list.entries ==
                __libecs_win32_pctx.mhdl_list.num_entries - 1) {
            __libecs_win32_pctx.mhdl_list.num_entries--;
        }

        libecs_win32_mmap_hdl_list_unlock(
                &__libecs_win32_pctx.mhdl_list);

        {
            BOOL r1 = UnmapViewOfFile(start);
            BOOL r2 = CloseHandle(mhdl);

            if (r1 && r2)
                return 0;
        }
    } else {
        if (VirtualFree(start, 0, MEM_RELEASE))
            return 0;
    }

    __libecs_errno = libecs_win32_translate_errno(GetLastError()); 

    return -1;
}

int libecs_win32_mman_sync(void *start, size_t length, int flags)
{
    if (FlushViewOfFile(start, length))
        return 0;

    __libecs_errno = libecs_win32_translate_errno(GetLastError()); 

    return -1;    
}
