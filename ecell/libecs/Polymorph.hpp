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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __POLYMORPH_HPP
#define __POLYMORPH_HPP

#include <boost/static_assert.hpp>

#include "dmtool/DMObject.hpp"
#include "libecs.hpp"
#include "Converters.hpp"
#include "Util.hpp"

/**
   @addtogroup polymorph The Polymorph.
   The Polymorph

   @ingroup libecs
 */

/** @{ */

/** @file */

namespace libecs {

using namespace converter;

class Polymorph;

typedef std::vector<Polymorph> PolymorphVector;


class LIBECS_API PolymorphValue
{
public:

    virtual ~PolymorphValue();

    virtual const String  asString()        const = 0;
    virtual const Real    asReal()          const = 0;
    virtual const Integer asInteger()       const = 0;
    virtual const PolymorphVector asPolymorphVector() const = 0;

    template< typename T >
    const T as() const;

    virtual PolymorphValue* createClone() const = 0;

protected:

    PolymorphValue( const PolymorphValue& ) {}
    PolymorphValue() {}

private:

    const Polymorph& operator= ( const Polymorph& );

};


template <>
inline const String PolymorphValue::as() const
{
    return asString();
}

template <>
inline const Real PolymorphValue::as() const
{
    return asReal();
}

template <>
inline const Integer PolymorphValue::as() const
{
    return asInteger();
}

template <>
inline const PolymorphVector PolymorphValue::as() const
{
    return asPolymorphVector();
}



template< typename T >
class LIBECS_API ConcretePolymorphValue
            :
            public PolymorphValue
{

    typedef typename libecs::Param<T>::type TParam;

public:

    ConcretePolymorphValue( TParam aValue )
            :
            value_( aValue )
    {
        ; // do nothing
    }

    ConcretePolymorphValue( const PolymorphValue& aValue )
            :
            value_( aValue.as<T>() )
    {
        ; // do nothing
    }

    virtual ~ConcretePolymorphValue()
    {
        ; // do nothing
    }

    virtual const String asString() const
    {
        return convertTo<String>( value_ );
    }

    virtual const Real   asReal()  const
    {
        return convertTo<Real>( value_ );
    }

    virtual const Integer asInteger()   const
    {
        return convertTo<Integer>( value_ );
    }

    virtual const PolymorphVector asPolymorphVector() const
    {
        return convertTo<PolymorphVector>( value_ );
    }

    virtual PolymorphValue* createClone() const
    {
        return new ConcretePolymorphValue<T>( *this );
    }

private:

    T value_;

};




class LIBECS_API PolymorphNoneValue
            :
            public PolymorphValue
{

public:

    PolymorphNoneValue() {}

    virtual ~PolymorphNoneValue();

    virtual const String  asString() const;
    virtual const Real    asReal() const       {
        return 0.0;
    }
    virtual const Integer asInteger() const    {
        return 0;
    }
    virtual const PolymorphVector asPolymorphVector() const;

    virtual PolymorphValue* createClone() const
    {
        return new PolymorphNoneValue;
    }

};



class LIBECS_API Polymorph
{

public:

    enum Type
    {
        NONE,
        REAL,
        INTEGER,
        STRING,
        POLYMORPH_VECTOR
    };


    Polymorph()
            :
            value_( new PolymorphNoneValue )
    {
        ; // do nothing
    }

    Polymorph( const String&  aValue )
            :
            value_( new ConcretePolymorphValue<String>( aValue ) )
    {
        ; // do nothing
    }

    Polymorph( RealParam aValue )
            :
            value_( new ConcretePolymorphValue<Real>( aValue ) )
    {
        ; // do nothing
    }

    Polymorph( IntegerParam aValue )
            :
            value_( new ConcretePolymorphValue<Integer>( aValue ) )
    {
        ; // do nothing
    }

    Polymorph( const PolymorphVector& aValue )
            :
            value_( new ConcretePolymorphValue<PolymorphVector>( aValue ) )
    {
        ; // do nothing
    }

    Polymorph( const Polymorph& aValue )
            :
            value_( aValue.createValueClone() )
    {
        ; // do nothing
    }

    ~Polymorph()
    {
        delete value_;
    }

    const Polymorph& operator=( const Polymorph& rhs )
    {
        if ( this != &rhs )
        {
            delete value_;
            value_ = rhs.createValueClone();
        }

        return *this;
    }

    const String asString() const
    {
        return value_->asString();
    }

    const Real   asReal() const
    {
        return value_->asReal();
    }

    const Integer asInteger() const
    {
        return value_->asInteger();
    }

    const PolymorphVector asPolymorphVector() const
    {
        return value_->asPolymorphVector();
    }

    template< typename T >
    const T as() const;
    //    {
    //      DefaultSpecializationInhibited();
    //    }

    const Type getType() const;

    void changeType( const Type aType );


    operator String() const
    {
        return asString();
    }

    operator Real() const
    {
        return asReal();
    }

    operator Integer() const
    {
        return asInteger();
    }

    operator PolymorphVector() const
    {
        return asPolymorphVector();
    }

protected:

    PolymorphValue* createValueClone() const
    {
        return value_->createClone();
    }

protected:

    PolymorphValue* value_;

};




template <>
inline const String Polymorph::as() const
{
    return asString();
}

template <>
inline const Real   Polymorph::as() const
{
    return asReal();
}

template <>
inline const Integer Polymorph::as() const
{
    return asInteger();
}

template <>
inline const PolymorphVector Polymorph::as() const
{
    return asPolymorphVector();
}

LIBECS_API const Polymorph convertStringMapToPolymorph( std::map<const String&, const String&> const& aMap );

/**
   nullValue() specialization for Polymorph. See Util.hpp
 */
template<>
inline const Polymorph nullValue()
{
    return Polymorph();
}

// convertTo template specializations for PolymorphVector.
namespace converter { namespace detail {
    template<>
    struct ConvertTo< PolymorphVector, std::map< const String, const String > >
    {
        typedef PolymorphVector ToType;
        typedef std::map< const String, const String > FromType;

        struct Converter
        {
            ToType operator()( const FromType& aMap )
            {
                PolymorphVector aVector;
                aVector.reserve( aMap.size() );

                for ( FromType::const_iterator i( aMap.begin() );
                        i != aMap.end();  ++i )
                {
                    PolymorphVector anInnerVector;
                    anInnerVector.push_back( i->first );
                    anInnerVector.push_back( i->second );

                    aVector.push_back( anInnerVector );
                }

                return aVector;
            }
        };
    };

    template< typename Tfrom_ >
    struct ConvertTo< PolymorphVector, Tfrom_ >
    {
        typedef PolymorphVector ToType;
        typedef Tfrom_ FromType;

        struct Converter
        {
            const PolymorphVector operator()( const FromType& aValue )
            {
                return PolymorphVector( 1, aValue );
            }
        };
    };

    // Override the <T,String> case defined in convertTo.hpp.
    template<>
    struct ConvertTo< PolymorphVector, String >
    {
        typedef PolymorphVector ToType;
        typedef String FromType;

        struct Converter
        {
            const PolymorphVector operator()( const String& aValue )
            {
                return PolymorphVector( 1, aValue );
            }
        };
    };

    // from PolymorphVector
    template<>
    struct ConvertTo< PolymorphVector, PolymorphVector >
    {
        typedef PolymorphVector ToType;
        typedef PolymorphVector FromType;

        typedef StaticCaster< PolymorphVector, PolymorphVector > Converter;
    };


    // from PolymorphVector
    template< typename Tto_ >
    struct ConvertTo< Tto_, PolymorphVector >
    {
        typedef Tto_ ToType;
        typedef PolymorphVector FromType;

        struct Converter
        {
            const ToType operator()( const PolymorphVector& aValue )
            {
                checkSequenceSize( aValue, 1 );
                return static_cast<Polymorph>( aValue[0] ).as<ToType>();
            }
        };
    };

    // Override the <String,T> case defined in convertTo.hpp.
    template<>
    struct ConvertTo< String, PolymorphVector >
    {
        typedef String ToType;
        typedef PolymorphVector FromType;

        struct Converter
        {
            const String operator()( const PolymorphVector& aValue )
            {
                checkSequenceSize( aValue, 1 );
                return static_cast<Polymorph>( aValue[0] ).as<String>();
            }
        };
    };
} } // namespace converter::detail

} // namespace libecs

/** @} */

#endif /* __POLYMORPH_HPP */
