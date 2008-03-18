//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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
// based on the code by Nathan Addy <addy@molsci.org>
//

#ifndef __HAPPENING_HPP
#define __HAPPENING_HPP

#include <utility>
#include <set>
#include <string.h>

#include "libecs.hpp"

namespace libecs
{

template<typename Toh_, typename Tdesc_>
class Happening
{
public:
    typedef Toh_ ObserverHandle;
    typedef typename ObserverHandle::element_type Observer;
    typedef void (Observer::*CallbackFunc)( const Tdesc_& );
    struct Subscription
    {
        ObserverHandle handle;
        CallbackFunc callback;

        Subscription( ObserverHandle _handle, CallbackFunc _callback )
            : handle( _handle ), callback( _callback ) {}

        bool operator<( const Subscription& that ) const
        {
            return handle < that.handle 
                || ( handle == that.handle && ::memcmp(
                    reinterpret_cast< const void* >( &callback ),
                    reinterpret_cast< const void* >( &that.callback ),
                    sizeof( CallbackFunc ) ) < 0 );
        } 
    };
protected:
    typedef ::std::set< Subscription > Subscriptions;

public:
    Happening()
    {
        ; // do nothing
    }

    ~Happening()
    {
    }

    void add( const Subscription& sub )
    {
        subscriptions.insert( sub );
    }

    void remove( const Subscription& sub )
    {
        subscriptions.erase( sub );
    }

    void operator()( const Tdesc_& desc )
    {
        bool interrupted( false );
        for ( typename Subscriptions::const_iterator i( subscriptions.begin() );
                i != subscriptions.end(); ++i)
        {
            try
            {
                ((*i->handle).*(i->callback))( desc );
            }
            catch ( const std::exception& exc )
            {
                interrupted = true;
            }
        }
        if ( interrupted )
        {
            THROW_EXCEPTION( Interruption,
                    "happening may not have been notified to all "
                    "the observers due to exception" );
        }
    }
protected:
    Subscriptions subscriptions;
};

}

#endif /* __HAPPENING_HPP */
