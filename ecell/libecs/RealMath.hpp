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
// written by
//     Koichi Takahashi <shafi@e-cell.org>,
//     Yasuhiro Naito
// E-Cell Project.
//

#ifndef __REALMATH_HPP
#define __REALMATH_HPP

#include <vector>
// #include <iostream>

namespace libecs { namespace math {

/**
   These functions are prepared for ExpressionFluxProcess
   and are used in it. asinh, acosh and atanh are not available in
   MS Windows (MinGW).
*/
template <typename T>
inline Real real_not( T n )
{
    if ( n == 0 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_eq( T n1, T n2 )
{
    if ( n1 == n2 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_neq( T n1, T n2 )
{
    if ( n1 == n2 )
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}

template <typename T>
inline Real real_gt( T n1, T n2 )
{
    if ( n1 > n2 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_lt( T n1, T n2 )
{
    if ( n1 < n2 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_geq( T n1, T n2 )
{
    if ( n1 >= n2 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_leq( T n1, T n2 )
{
    if ( n1 <= n2 )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_and( T n1, T n2 )
{
    if ( ( n1 != 0 ) && ( n2 != 0 ) )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_or( T n1, T n2 )
{
    if ( ( n1 != 0 ) || ( n2 != 0 ) )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline Real real_xor( T n1, T n2 )
{
    if ( ( n1 != 0 ) && !( n2 != 0 ) )
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

template <typename T>
inline T asinh( T n )
{
    return log( n + sqrt( n * n + 1 ) );
}

template <typename T>
inline T acosh( T n )
{
    return log( n - sqrt( n * n - 1 ) );
}

template <typename T>
inline T atanh( T n )
{
    return 0.5 * log( ( 1 + n ) / ( 1 - n ) );
}

template <typename T>
inline T sec( T n )
{
    return 1 / cos( n );
}

template <typename T>
inline T csc( T n )
{
    return 1 / sin( n );
}

template <typename T>
inline T cot( T n )
{
    return 1 / tan( n );
}

template <typename T>
inline T asec( T n )
{
    return 1 / acos( n );
}

template <typename T>
inline T acsc( T n )
{
    return 1 / asin( n );
}

template <typename T>
inline T acot( T n )
{
    return 1 / atan( n );
}

template <typename T>
inline T sech( T n )
{
    return 1 / cosh( n );
}

template <typename T>
inline T csch( T n )
{
    return 1 / sinh( n );
}

template <typename T>
inline T coth( T n )
{
    return 1 / tanh( n );
}

template <typename T>
inline T asech( T n )
{
    return 1 / acosh( n );
}

template <typename T>
inline T acsch( T n )
{
    return 1 / asinh( n );
}

template <typename T>
inline T acoth( T n )
{
    return 1 / atanh( n );
}

template <typename T>
inline T fact( T n )
{
    if ( n <= 1 )
        return 1;
    else
        return n * fact( n-1 );
}

template <typename T>
inline Real piecewise( std::vector<T> p )
{
    // std::cout << "Call Piecewise Function:" << std::endl;
    typename std::vector<T>::reverse_iterator pi = p.rbegin();
    while ( pi != p.rend() )
    {
        // std::cout << "  Value    : " << *pi << std::endl;
        pi++;
        // std::cout << "  Condition: " << *pi << std::endl;
        if ( pi == p.rend() || *pi != 0.0 ) return *(--pi);
        
        else if ( ++pi == p.rend() )
        {
            THROW_EXCEPTION( UnexpectedError,
                             "piecewise function couldn't determine suitable subdomain" );
        }
    }
        THROW_EXCEPTION( UnexpectedError,
                         "piecewise function has no argument." );
}

template <typename T>
inline Real delay( T n1, T n2, T n3 )
{
    // This method is never used. Just a dummy.
    return 0.0;
}

} } // namespace libecs::math

#endif /* __REALMATH_HPP */
