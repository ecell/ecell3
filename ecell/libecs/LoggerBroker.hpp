//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//


#if !defined(__LOGGER_BROKER_HPP)
#define __LOGGER_BROKER_HPP

#include <map>
#include <utility>
#include <boost/noncopyable.hpp>
#include <boost/mpl/if.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "libecs/Defs.hpp"
#include "libecs/FullID.hpp"
#include "libecs/Logger.hpp"

namespace libecs
{
// forward declaration
class Model;

/**
   LoggerBroker creates and administrates Loggers in a model.

   This class creates, holds in a map which associates FullPN with a Logger,
   and responds to requests to Loggers.

   @see FullPN
   @see Logger
*/
class LIBECS_API LoggerBroker
{
    friend class Entity;
public:
    typedef std::map< const FullID, std::map< String, Logger* >,
                      std::less< const FullID > > LoggerMap;

    typedef LoggerMap::value_type::second_type PerFullIDMap;

    template< typename Tderived_, typename Tconstness_ >
    struct iterator_base
    {
        typedef std::bidirectional_iterator_tag iterator_category;
        typedef std::ptrdiff_t difference_type;
        typedef std::pair<
            FullPN const,
            typename boost::mpl::if_<
                Tconstness_,
                typename boost::add_const< Logger >::type,
                Logger >::type* > value_type;
        typedef void pointer;
        typedef std::pair<
            FullPN const,
            typename boost::mpl::if_<
                Tconstness_,
                typename boost::add_const< Logger >::type,
                Logger >::type* > reference;

        typedef typename boost::mpl::if_< Tconstness_,
            typename LoggerMap::const_iterator,
            typename LoggerMap::iterator >::type
            outer_iterator_type;
        typedef typename boost::mpl::if_< Tconstness_,
            typename outer_iterator_type::value_type::second_type::const_iterator,
            typename outer_iterator_type::value_type::second_type::iterator >::type
            inner_iterator_type;
        typedef std::pair< outer_iterator_type, inner_iterator_type >
            iterator_pair;

        typedef iterator_base base_type;
 
        template< typename Trange_ > 
        iterator_base( iterator_pair const& aPair, Trange_& anOuterRange,
                       inner_iterator_type const& aNullInnerIterator )
            : thePair( aPair ), theOuterRange( boost::begin( anOuterRange ),
                                               boost::end( anOuterRange )),
              theNullInnerIterator( aNullInnerIterator ) {}

        Tderived_& operator++()
        {
            if ( theNullInnerIterator == thePair.second )
            {
                if ( theOuterRange.begin() == thePair.first &&
                     theOuterRange.begin() != theOuterRange.end() )
                {
                    thePair.second = thePair.first->second.begin();
                }
            }
            else
            {
                ++thePair.second;
                if ( thePair.first->second.end() == thePair.second )
                {
                    ++thePair.first;
                    if ( theOuterRange.end() == thePair.first )
                    {
                        thePair.second = theNullInnerIterator;
                    }
                    else
                    {
                        thePair.second = thePair.first->second.begin();
                    }
                }
            }

            return *static_cast< Tderived_* >( this );
        }

        Tderived_ operator++(int)
        {
            Tderived_ retval( *this );
            ++(*this);
            return retval;
        }

        Tderived_& operator--()
        {
            if ( thePair.first->second.begin() == thePair.second
                 || theOuterRange.end() == thePair.first )
            {
                if ( theOuterRange.begin() != thePair.first )
                {
                    --thePair.first;
                    thePair.second = thePair.first->second.end();
                    --thePair.second;
                }
                else
                {
                    thePair.second = theNullInnerIterator;
                }
            }
            else
            {
                --thePair.second;
            }

            return *static_cast< Tderived_* >( this );
        }

        Tderived_ operator--(int)
        {
            Tderived_ retval( *this );
            --(*this);
            return retval;
        }

        template< typename T_ >
        bool operator==(T_ const& rhs) const
        {
            return thePair.first == rhs.thePair.first &&
                    ( thePair.first == theOuterRange.end()
                      || thePair.second == rhs.thePair.second );
        }

        template< typename T_ >
        bool operator!=(T_ const& rhs) const
        {
            return !operator==(rhs);
        }

        reference operator*()
        {
            return std::make_pair(
                FullPN( thePair.first->first, thePair.second->first ), 
                thePair.second->second );
        }

    public:
        iterator_pair thePair;
        boost::iterator_range< outer_iterator_type > theOuterRange;
        inner_iterator_type theNullInnerIterator;
    };

    struct iterator
        : iterator_base< iterator, boost::mpl::bool_< false > >
    {
        iterator( base_type const& that ): base_type( that ) {}

        template< typename Trange_ >
        iterator( iterator_pair const& aPair, Trange_& anOuterRange,
                  inner_iterator_type const& aNullInnerIterator )
            : base_type( aPair, anOuterRange, aNullInnerIterator ) {}
    };

    struct const_iterator
        : iterator_base< const_iterator, boost::mpl::bool_< true > >
    {
        const_iterator( base_type const& that ): base_type( that ) {}

        const_iterator( iterator const& that )
            : base_type( that.thePair, that.theOuterRange,
                         that.theNullInnerIterator ) {}

        template< typename Trange_ >
        const_iterator( iterator_pair const& aPair,
                        Trange_ const& anOuterRange,
                        inner_iterator_type const& aNullInnerIterator )
            : base_type( aPair, anOuterRange, aNullInnerIterator ) {}
    };

    typedef boost::transform_iterator<
        SelectSecond< PerFullIDMap::iterator::value_type >,
        PerFullIDMap::iterator > PerFullIDLoggerIterator;

    typedef boost::transform_iterator<
        SelectSecond< PerFullIDMap::const_iterator::value_type >,
        PerFullIDMap::const_iterator > PerFullIDLoggerConstIterator;

    typedef boost::iterator_range< PerFullIDLoggerIterator > LoggersPerFullID;

    typedef boost::iterator_range< PerFullIDLoggerConstIterator > ConstLoggersPerFullID;

public:
    LoggerBroker( Model const& aModel );

    ~LoggerBroker();

    Model const& getModel() const
    {
        return theModel;
    }

    /**
       Get or create a Logger for a PropertySlot.

       This method first look for a Logger object which is logging
       the specified PropertySlot, and if it is found, returns the
       Logger.    If there is no Logger connected to the PropertySlot yet,
       it creates and returns a new Logger.    

       FIXME: doc for interval needed

       @param aFullPN         a FullPN of the requested FullPN
       @param anInterval    a logging interval
       @return a borrowed pointer to the Logger
       
    */

    Logger* getLogger( FullPN const& aFullPN ) const;

    Logger* createLogger( FullPN const& aFullPN, Logger::Policy const& aParamList );

    void removeLogger( FullPN const& aFullPN );

    /**
       Flush the data in all the Loggers immediately.

       Usually Loggers record data with logging intervals.    This method
       orders every Logger to write the data immediately ignoring the
       logging interval.
   
    */
    void flush();

    iterator begin()
    {
        return iterator(
            iterator::iterator_pair(
                theLoggerMap.begin(),
                theLoggerMap.begin() == theLoggerMap.end() ?
                    theEmptyPerFullIDMap.begin():
                    theLoggerMap.begin()->second.begin() ),
            theLoggerMap, theEmptyPerFullIDMap.begin() );
    }

    iterator end()
    {
        return iterator(
            iterator::iterator_pair(
                theLoggerMap.end(),
                theLoggerMap.begin() == theLoggerMap.end() ?
                    theEmptyPerFullIDMap.begin():
                    theLoggerMap.begin()->second.end() ),
            theLoggerMap, theEmptyPerFullIDMap.begin() );
    }

    const_iterator begin() const
    {
        return const_iterator(
            const_iterator::iterator_pair(
                theLoggerMap.begin(),
                theLoggerMap.begin() == theLoggerMap.end() ?
                    theEmptyPerFullIDMap.begin():
                    theLoggerMap.begin()->second.begin() ),
            theLoggerMap, theEmptyPerFullIDMap.begin() );
    }

    const_iterator end() const
    {
        return const_iterator(
            const_iterator::iterator_pair(
                theLoggerMap.end(),
                theLoggerMap.begin() == theLoggerMap.end() ?
                    theEmptyPerFullIDMap.begin():
                    theLoggerMap.begin()->second.end() ),
            theLoggerMap, theEmptyPerFullIDMap.begin() );
    }

    LoggersPerFullID getLoggersByFullID( FullID const& aFullID );

    ConstLoggersPerFullID getLoggersByFullID( FullID const& aFullID ) const;

    void removeLoggersByFullID( FullID const& aFullID );

private:
    /// non-copyable
    LoggerBroker( LoggerBroker const& );

private:
    LoggerMap     theLoggerMap;
    PerFullIDMap  theEmptyPerFullIDMap;
    Model const&  theModel;
};

} // namespace libecs

#endif /* __LOGGER_BROKER_HPP */
