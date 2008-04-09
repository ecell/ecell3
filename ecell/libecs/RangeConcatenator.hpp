#ifndef BOOST_PP_IS_ITERATING

#ifndef __RANGE_CONCATENATOR_HPP
#define __RANGE_CONCATENATOR_HPP

#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/comparison/greater.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/limits.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/iterator.hpp>
#include <boost/range/value_type.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/type_traits.hpp>
#include <boost/variant.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/fold.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/size.hpp>
#include <boost/mpl/lambda.hpp>
#include <boost/mpl/assert.hpp>

#define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, "libecs/RangeConcatenator.hpp"))

namespace libecs
{

template<typename TMseq_,
    typename Tval_ = typename ::boost::range_value<
            typename ::boost::mpl::at_c< TMseq_, 0 >::type >::type,
    long TMsize_ = boost::mpl::size<TMseq_>::value >
struct RangeConcatenator 
{
};

#include BOOST_PP_ITERATE()

} // namespace libecs

#endif /* __RANGE_CONCATENATOR_HPP */
#else
    

#define RANGE_CONCATENATOR_MPL_ENUM_TPL(__z__, __n__, __d__) \
    typename ::boost::mpl::at_c<__d__, __n__>::type&

#define RANGE_CONCATENATOR_MPL_ENUM_ITERATOR_TPL(__z__, __n__, __d__) \
    typename ::boost::range_iterator< typename ::boost::mpl::at_c<__d__, __n__>::type >::type

#define RANGE_CONCATENATOR_MPL_ENUM_CONST_ITERATOR_TPL(__z__, __n__, __d__) \
    typename ::boost::range_iterator< const typename ::boost::mpl::at_c<__d__, __n__>::type >::type

#define RANGE_CONCATENATOR_RANGE_TUPLE_SIZE_SUM_TPL(__z__, __n__, __d__) \
    BOOST_PP_IF(BOOST_PP_GREATER(__n__, 0), +, ) ::boost::size( range<__n__>() )

#define RANGE_CONCATENATOR_RANGE_TUPLE_LOOKUP_TPL(__z__, __n__, __d__) \
    if ( __d__ >= ::boost::size( range<__n__>() ) ) \
        __d__ -=  ::boost::size( range<__n__>() ); \
    else \
        return *offsetting( ::boost::begin( range<__n__>() ), __d__ );

#define RANGE_CONCATENATOR_RANGE_SELECT_TPL(__z__, __n__, __d__) \
    case __n__:\
        return __d__( range<__n__>() );

#define RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_TPL(__z__, __n__, __d__) \
    case __n__:\
        return *(::boost::get< typename ::boost::mpl::at_c< \
                BOOST_PP_TUPLE_ELEM( 2, 0, __d__ ), __n__ >::type >( \
                BOOST_PP_TUPLE_ELEM( 2, 1, __d__ ) ) );

#define RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_DIFF_TPL(__z__, __n__, __d__) \
    case __n__:\
        { \
            typedef typename ::boost::mpl::at_c< \
                    BOOST_PP_TUPLE_ELEM( 3, 0, __d__ ), __n__ >::type type; \
            return calc_distance( ::boost::get< type >( \
                        BOOST_PP_TUPLE_ELEM( 3, 1, __d__ ) ), \
                    ::boost::get< type >( \
                        BOOST_PP_TUPLE_ELEM( 3, 2, __d__ ) ) ); \
        }

#define RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_ADD_TPL(__z__, __n__, __d__) \
    case __n__:\
        return offsetting( ::boost::get< typename ::boost::mpl::at_c< \
                BOOST_PP_TUPLE_ELEM( 3, 0, __d__ ), __n__ >::type>( \
                    BOOST_PP_TUPLE_ELEM( 3, 1, __d__ ) ), \
                BOOST_PP_TUPLE_ELEM( 3, 2, __d__ ) ); \

#define RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_DEREF_TPL(__z__, __n__, __d__) \
    case __n__:\
        return *offsetting( ::boost::get< typename ::boost::mpl::at_c< \
                BOOST_PP_TUPLE_ELEM( 3, 0, __d__ ), __n__ >::type>( \
                    BOOST_PP_TUPLE_ELEM( 3, 1, __d__ ) ), \
                BOOST_PP_TUPLE_ELEM( 3, 2, __d__ ) ); \

#define RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_UNARY_OP_TPL(__z__, __n__, __d__) \
    case __n__:\
        BOOST_PP_TUPLE_ELEM( 3, 2, __d__ ) \
        ::boost::get< typename ::boost::mpl::at_c< \
                BOOST_PP_TUPLE_ELEM( 3, 0, __d__ ), __n__ >::type>( \
                    BOOST_PP_TUPLE_ELEM( 3, 1, __d__ ) ); \
        break;

template< typename TMseq_, typename Tval_ >
class RangeConcatenator< TMseq_, Tval_, BOOST_PP_ITERATION() >
{
private:
    template< typename T1_, typename T2_ >
    struct whether_value_type_is_convertible
    {
        typedef ::boost::mpl::and_<
                ::boost::is_convertible<
                    typename ::boost::range_value< T2_ >::type,
                    Tval_ >,
                T1_ > type;
    };

    BOOST_MPL_ASSERT(( ::boost::mpl::fold< TMseq_, ::boost::mpl::true_,
            ::boost::mpl::quote2< whether_value_type_is_convertible > > ));

    template< typename Titer_, typename TiterCat_ >
    class offsetter
    {
    public:
        offsetter( const Titer_& iter ): iter_( iter ) {}

        Titer_ operator()(
            typename ::boost::iterator_difference< Titer_ >::type offset )
        {
            Titer_ retval( iter_ );
            while ( offset > 0 )
            {
                --offset;
                ++retval;
            }
            while ( offset < 0 )
            {
                ++offset;
                --retval;
            }
            return retval;
        }

    private:
        Titer_ iter_;
    };

    template<typename Titer_>
    class offsetter<Titer_, ::std::random_access_iterator_tag >
    {
    public:
        offsetter( const Titer_& iter ): iter_( iter ) {}

        Titer_ operator()(
            typename ::boost::iterator_difference< Titer_ >::type offset )
        {
            return iter_ + offset;
        }

    private:
        Titer_ iter_;
    };


    template<typename Titer_>
    static Titer_ offsetting( const Titer_& iter,
            typename ::boost::iterator_difference< Titer_ >::type offset )
    {
        return offsetter< Titer_,
            typename ::boost::iterator_category< Titer_ >::type >( iter )(
                offset );
    }

    template< typename Titer_, typename TiterCat_ >
    class distance_calculator
    {
    public:
        distance_calculator( const Titer_& iter ): iter_( iter ) {}

        typename ::boost::iterator_difference< Titer_ >::type operator()(
                const Titer_& that )
        {
            typename Titer_::difference_type retval( 0 );
            Titer_ f( that ), b( that );
            for ( ;; )
            {
                if ( f == iter_ )
                {
                    retval = -retval;
                    break;
                }
                else if ( b == iter_ )
                {
                    break;
                }
                ++f, --b;
                ++retval;
            }

            return retval;
        }

    private:
        Titer_ iter_;
    };

    template<typename Titer_>
    class distance_calculator<Titer_, ::std::random_access_iterator_tag >
    {
    public:
        distance_calculator( const Titer_& iter ): iter_( iter ) {}

        typename ::boost::iterator_difference< Titer_ >::type operator()(
                const Titer_& that )
        {
            return iter_ - that;
        }

    private:
        Titer_ iter_;
    };

    template<typename Titer_>
    static typename ::boost::iterator_difference< Titer_ >::type
    calc_distance( const Titer_& lhs, const Titer_& rhs )
    {
        return distance_calculator< Titer_, typename boost::iterator_category< Titer_ >::type >( lhs )( rhs );
    }

public:
    typedef RangeConcatenator self_type;
    typedef TMseq_ range_type_list_type;
    typedef typename ::boost::mpl::if_<
            ::boost::is_same<
                typename ::boost::range_value< typename ::boost::mpl::at_c<
                        TMseq_, 0 >::type >::type,
                Tval_>, Tval_&, Tval_>::type return_value_type;
    typedef Tval_ value_type;
    typedef value_type& reference;
    typedef value_type* pointer;
    typedef ::std::size_t size_type;
    typedef ::std::ptrdiff_t difference_type;
    typedef ::boost::tuple<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_TPL,
                range_type_list_type
            )
        > range_list_type;
    typedef ::boost::variant<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_TPL,
                range_type_list_type
            )
        > range_variant_type;
    typedef ::boost::variant<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_ITERATOR_TPL,
                range_type_list_type
            )
        > iterator_variant_type;
    typedef ::boost::variant<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_CONST_ITERATOR_TPL,
                range_type_list_type
            )
        > const_iterator_variant_type;
    typedef ::boost::mpl::vector<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_ITERATOR_TPL,
                range_type_list_type
            )
        > iterator_type_list_type;
    typedef ::boost::mpl::vector<
            BOOST_PP_ENUM(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_MPL_ENUM_CONST_ITERATOR_TPL,
                range_type_list_type
            )
        > const_iterator_type_list_type;

protected:
    template< typename Tiv_, typename Til_ >
    class iterator_info
    {
    public:
        typedef ::std::ptrdiff_t difference_type;
        typedef std::bidirectional_iterator_tag iterator_category;

    public:
        iterator_info( RangeConcatenator& impl,
                ::std::size_t range_idx, Tiv_ iter )
            : impl_( impl ), range_idx_( range_idx ), iter_( iter )
        {
        }

        iterator_info( RangeConcatenator& impl,
                ::std::size_t range_idx )
            : impl_( impl ), range_idx_( range_idx ),
              iter_( impl.begin( range_idx ) )
        {
        }

        bool operator<( const iterator_info& rhs ) const
        {
            return ( range_idx_ < rhs.range_idx_ )
                || ( range_idx_ == rhs.range_idx_ &&
                    distance( range_idx_, iter_, rhs.iter_ ) < 0 );
        }

        bool operator>( const iterator_info& rhs ) const
        {
            return ( range_idx_ > rhs.range_idx_ )
                || ( range_idx_ == rhs.range_idx_ &&
                    distance( range_idx_, iter_, rhs.iter_ ) > 0 );
        }

        bool operator==( const iterator_info& rhs ) const
        {
            return ( range_idx_ == rhs.range_idx_ && iter_ == rhs.iter_ );
        }

        bool operator!=( const iterator_info& rhs ) const
        {
            return !operator==( rhs );
        }

        bool operator>=( const iterator_info& rhs ) const
        {
            return !operator<( rhs );
        }

        bool operator<=( const iterator_info& rhs ) const
        {
            return !operator>( rhs );
        }

        return_value_type operator*() const
        {
            switch ( range_idx_ ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_TPL,
                    ( Til_, iter_ )
                )
            }
        }

        return_value_type operator[]( difference_type off ) const
        {
            return *(*this + off);
        }

        static difference_type distance(
                ::std::size_t range_idx,
                const Tiv_& lhs, const Tiv_& rhs )
        {
            switch ( range_idx ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_DIFF_TPL,
                    ( Til_, lhs, rhs )
                )
            }
        } 

        static Tiv_ add(
                ::std::size_t range_idx,
                const Tiv_& lhs, difference_type rhs )
        {
            switch ( range_idx ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_ADD_TPL,
                    ( Til_, lhs, rhs )
                )
            }
        } 

        static void increment( ::std::size_t range_idx, Tiv_& iter )
        {
            switch ( range_idx ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_UNARY_OP_TPL,
                    ( Til_, iter, ++ )
                )
            }
        }

        static void decrement( ::std::size_t range_idx, Tiv_& iter )
        {
            switch ( range_idx ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_UNARY_OP_TPL,
                    ( Til_, iter, -- )
                )
            }
        }

        iterator_info operator+( difference_type off ) const
        {
            difference_type o( off );
            iterator_info retval( *this );
            Tiv_ begin( impl_.begin( retval.range_idx_ ) );
            Tiv_ end( impl_.end( retval.range_idx_ ) );

            for (;;)
            {
                retval.iter_ = add( retval.range_idx_, retval.iter_, o );
                if ( distance( retval.range_idx_, retval.iter_, begin ) < 0 )
                {
                    if ( retval.range_idx_ == 0 )
                    {
                        throw ::std::out_of_range("offset out of range: "
                                + boost::lexical_cast< ::std::string>( off ) );
                    }
                    o = distance( retval.range_idx_, retval.iter_, begin );
                    --retval.range_idx_;
                    begin = impl_.begin( retval.range_idx_ );
                    end = impl_.end( retval.range_idx_ );
                    retval.iter_ = end;
                }
                else if ( distance( retval.range_idx_, retval.iter_, end ) >= 0 )
                {
                    if ( retval.range_idx_ >= impl_.range_count() )
                    {
                        throw ::std::out_of_range("offset out of range: "
                                + boost::lexical_cast< ::std::string>( off ) );
                    }
                    o = distance( retval.range_idx_, retval.iter_, end );
                    ++retval.range_idx_;
                    begin = impl_.begin( retval.range_idx_ );
                    end = impl_.end( retval.range_idx_ );
                    retval.iter_ = begin;
                }
                else
                {
                    break;
                }
            }
            return retval;
        }

        difference_type operator-( const iterator_info& rhs ) const
        {
            difference_type retval = 0;
            Tiv_ i( iter_ );
            ::std::size_t ridx( range_idx_ );
            Tiv_ begin( impl_.begin( ridx ) );
            Tiv_ end( impl_.end( ridx ) );

            if ( rhs.range_idx_ > range_idx_ )
            {
                do
                {
                    retval -= distance( ridx, end, i );
                    ++ridx;
                    BOOST_ASSERT( ridx < impl_.range_count() );
                    begin = impl_.begin( ridx );
                    end = impl_.end( ridx );
                    i = begin;
                } while ( rhs.range_idx_ > ridx );
                retval -= rhs.iter_ - i;
            }
            else if ( rhs.range_idx_ < range_idx_ )
            {
                do
                {
                    retval += distance( ridx, i, begin );
                    --ridx;
                    BOOST_ASSERT( ridx >= 0 );
                    begin = impl_.begin( ridx );
                    end = impl_.end( ridx );
                    i = end;
                } while ( rhs.range_idx_ < ridx );
                retval += i - rhs.iter_;
            }
            else
            {
                retval = i - rhs.iter_;
            }
            return retval;
        }
    public:
        RangeConcatenator& impl_;
        ::std::size_t range_idx_;
        Tiv_ iter_;
    };

public:
    template< typename Tiv_, typename Til_ >
    class iterator_base: public iterator_info< Tiv_, Til_ >
    {
    public:
        typedef iterator_info< Tiv_, Til_ > base_type;
        typedef typename RangeConcatenator::value_type value_type;
        typedef typename RangeConcatenator::difference_type difference_type;
        typedef typename RangeConcatenator::pointer pointer;
        typedef typename RangeConcatenator::reference reference;
        typedef ::std::bidirectional_iterator_tag iterator_cateory;

    public:
        const iterator_base& operator++()
        {
            increment( base_type::range_idx_, base_type::iter_ );
            if ( base_type::iter_ == end_ )
            {
                if ( base_type::range_idx_ < base_type::impl_.range_count() - 1 )
                {
                    ++base_type::range_idx_;
                    base_type::iter_ = begin_ = base_type::impl_.begin(
                            base_type::range_idx_ );
                    end_ = base_type::impl_.end( base_type::range_idx_ );
                }
            }

            return *this;
        }

        const iterator_base& operator++(int)
        {
            iterator retval( *this, base_type::iter_ );

            increment( base_type::range_idx_, base_type::iter_ );
            if ( base_type::iter_ == end_ )
            {
                if ( base_type::range_idx_ < base_type::impl_.range_count() - 1 )
                {
                    ++base_type::range_idx_;
                    base_type::iter_ = begin_ = base_type::impl_.begin(
                            base_type::range_idx_ );
                    end_ = base_type::impl_.end( base_type::range_idx_ );
                }
            }

            return retval;
        }

        const iterator_base& operator--()
        {
            if ( base_type::iter_ == begin_ )
            {
                if ( base_type::range_idx_ > 0 )
                {
                   --base_type::range_idx_;
                    base_type::iter_ = end_ = base_type::impl_.end(
                            base_type::range_idx_ );
                    begin_ = base_type::impl_.begin( base_type::range_idx_ );
                }
            }

            decrement( base_type::range_idx_, base_type::iter_ );

            return *this;
        }

        const iterator_base& operator--(int)
        {
            iterator retval( *this, base_type::iter_ );

            if ( base_type::iter_ == begin_ )
            {
                if ( base_type::range_idx_ > 0 )
                {
                    --base_type::range_idx_;
                    base_type::iter_ = end_ = base_type::impl_.end( base_type::range_idx_ );
                    begin_ = base_type::impl_.begin( base_type::range_idx_ );
                }
            }

            decrement( base_type::range_idx_, base_type::iter_ );

            return retval;
        }

        reference operator[]( difference_type off )
        {
            if ( distance( base_type::range_idx_, base_type::iter_, begin_ ) + off >= 0 &&
                    distance( base_type::range_idx_, end_, base_type::iter_ ) > off )
            {
                switch ( base_type::range_idx_ ) {
                BOOST_PP_REPEAT(
                    BOOST_PP_ITERATION(),
                    RANGE_CONCATENATOR_RANGE_SELECT_ITER_VARIANT_DEREF_TPL,
                    ( Til_, base_type::iter_, off )
                )
                }
            }
            return base_type::operator[]( off );
        }

    protected:
        iterator_base( const iterator_info< Tiv_, Til_ >& info )
            : base_type( info ),
              begin_( info.impl_.begin( info.range_idx_ ) ),
              end_( info.impl_.end( info.range_idx_ ) ) {}

        iterator_base( const iterator_base& proto, const Tiv_& iter )
            : base_type( proto.impl_, proto.range_idx_, iter ),
              begin_( proto.begin_ ), end_( proto.end_ ) {}

    protected:
        Tiv_ begin_, end_;
    };

    struct iterator: public iterator_base< iterator_variant_type,
            iterator_type_list_type >
    {
        typedef iterator_base< iterator_variant_type,
                iterator_type_list_type > base_type;
        typedef typename base_type::base_type info_type;

        iterator( const info_type& that )
            : base_type( that ) {}
        
        iterator( const info_type& that, const iterator_variant_type& iter )
            : base_type( that, iter ) {}
    };

    friend struct iterator;

    struct const_iterator: public iterator_base< const_iterator_variant_type,
            const_iterator_type_list_type >
    {
        typedef iterator_base< const_iterator_variant_type,
                const_iterator_type_list_type > base_type;
        typedef typename base_type::base_type info_type;

        const_iterator( const info_type& that )
            : base_type( that ) {}
        
        const_iterator( const info_type& that,
                const const_iterator_variant_type& iter )
            : base_type( that, iter ) {}
    };

    friend struct const_iterator;

public:
    RangeConcatenator( const range_list_type& ranges )
        : ranges_( ranges ) {}

    size_type size() const
    {
        return BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_TUPLE_SIZE_SUM_TPL,);
    }

    template<long Nidx_>
    typename ::boost::mpl::at_c<range_type_list_type, Nidx_>::type& range()
    {
        return ::boost::get<Nidx_>( ranges_ );
    }

    template<long Nidx_>
    const typename ::boost::mpl::at_c<range_type_list_type, Nidx_>::type&
    range() const
    {
        return ::boost::get<Nidx_>( ranges_ );
    }

    range_variant_type range( ::std::size_t idx )
    {
        switch ( idx ) {
            BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_SELECT_TPL,
                ::boost::begin
            )
        }
    }

    iterator_variant_type begin( ::std::size_t idx )
    {
        switch ( idx ) {
            BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_SELECT_TPL,
                ::boost::begin
            )
        }
    }

    const_iterator_variant_type begin( ::std::size_t idx ) const
    {
        switch (idx) {
            BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_SELECT_TPL,
                ::boost::begin
            )
        }
    }

    iterator_variant_type end( ::std::size_t idx )
    {
        switch ( idx ) {
            BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_SELECT_TPL,
                ::boost::end
            )
        }
    }

    const_iterator_variant_type end( ::std::size_t idx ) const
    {
        switch (idx) {
            BOOST_PP_REPEAT(
                BOOST_PP_ITERATION(),
                RANGE_CONCATENATOR_RANGE_SELECT_TPL,
                ::boost::end
            )
        }
    }

    iterator begin()
    {
        return iterator( typename iterator::info_type( *this, 0 ) );
    }

    iterator end()
    {
        return iterator(
                typename iterator::info_type( *this, num_ranges_ - 1,
                    ::boost::get< num_ranges_ - 1 >( ranges_ ).end() ) );
    }

    return_value_type at( size_type idx )
    {
        BOOST_PP_REPEAT(
            BOOST_PP_ITERATION(),
            RANGE_CONCATENATOR_RANGE_TUPLE_LOOKUP_TPL,
            idx );
    }

    const return_value_type at( size_type idx ) const
    {
        BOOST_PP_REPEAT(
            BOOST_PP_ITERATION(),
            RANGE_CONCATENATOR_RANGE_TUPLE_LOOKUP_TPL,
            idx );
    }

    static ::std::size_t range_count()
    {
        return num_ranges_;
    }

    return_value_type operator[]( size_type idx )
    {
        return at( idx );
    }

    const return_value_type operator[]( size_type idx ) const
    {
        return at( idx );
    }

private:
    range_list_type ranges_;
    static const long num_ranges_ = BOOST_PP_ITERATION();
};

#endif /* BOOST_PP_ITERATING */
