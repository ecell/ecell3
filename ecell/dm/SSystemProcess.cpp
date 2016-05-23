//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2016 Keio University
//       Copyright (C) 2008-2016 RIKEN
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

#include <gsl/gsl_sf.h>
#include <boost/multi_array.hpp>

#include <libecs/libecs.hpp>
#include <libecs/ContinuousProcess.hpp>

#include "ESSYNSProcessInterface.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_EXTRA_1( SSystemProcess, ContinuousProcess,
                         ESSYNSProcessInterface )
{
public:
    LIBECS_DM_OBJECT( SSystemProcess, Process )
    {
        INHERIT_PROPERTIES( ContinuousProcess );

        PROPERTYSLOT_SET_GET( Integer, Order );
        PROPERTYSLOT_SET_GET( Polymorph, SSystemMatrix );
    }


    SSystemProcess()
        : theSystemSize( 0 ), Order( 3 )
    {
        ; // do nothing
    }

    virtual ~SSystemProcess()
    {
        ;
    }

    SIMPLE_GET_METHOD( Integer, Order );

    SET_METHOD( Integer, Order )
    {
        Order = value;

        // init Substance Vector
        theY.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );

        // init S-System Vector & Matrix
        theAlpha.resize( boost::extents[ theSystemSize + 1 ] );
        theBeta.resize( boost::extents[ theSystemSize + 1] );
        theG.resize( boost::extents[ theSystemSize + 1 ][ theSystemSize + 1 ] );
        theH.resize( boost::extents[ theSystemSize + 1 ][ theSystemSize + 1 ] );

        // init S-System tmp Vector & Matrix
        theAlphaBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theBetaBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theGBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theHBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );

        theFBuffer.resize( boost::extents[ Order + 1 ][ Order + 1 ] );
    }

    SET_METHOD( Polymorph, SSystemMatrix )
    {
        SSystemMatrix = value;
        PolymorphVector aValueVector( value.as<PolymorphVector>() );
        theSystemSize = aValueVector.size();

        // init Substance Vector
        theY.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );

        // init S-System Vector & Matrix
        theAlpha.resize( boost::extents[ theSystemSize + 1 ] );
        theBeta.resize( boost::extents[ theSystemSize + 1] );
        theG.resize( boost::extents[ theSystemSize + 1 ][ theSystemSize + 1 ] );
        theH.resize( boost::extents[ theSystemSize + 1 ][ theSystemSize + 1 ] );

        // init S-System tmp Vector & Matrix
        theAlphaBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theBetaBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theGBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );
        theHBuffer.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );

        theFBuffer.resize( boost::extents[ Order + 1 ][ Order + 1 ] );

        // init Factorial matrix
        for(int m( 2 ) ; m < Order+1 ; m++)
        {
            for(int q( 1 ); q < m ; q++)
            {
                const Real aFact( 1 / gsl_sf_fact(q-1) * gsl_sf_fact(m-q-1) * m * (m-1) );
                (theFBuffer[m])[q] = aFact;
            }
        }

        // set Alpha, Beta, G, H
        for( int i( 0 ); i < theSystemSize; ++i )
        {

            theAlpha[i+1] = (aValueVector[i].as<PolymorphVector>())[0].as<Real>() ;
            for( int j( 0 ); j < theSystemSize; ++j )
            {
                if( i == j )
                {

                    (theG[i+1])[j+1] = (aValueVector[i].as<PolymorphVector>())[j+1].as<Real>() - 1 ;
                }
                else
                {
                    (theG[i+1])[j+1] = (aValueVector[i].as<PolymorphVector>())[j+1].as<Real>() ;
                }
            }
            theBeta[i+1] = (aValueVector[i].as<PolymorphVector>())[1+theSystemSize].as<Real>() ;
            for( int j( 0 ); j < theSystemSize; ++j )
            {
                if( i == j )
                {
                    (theH[i+1])[j+1] = (aValueVector[i].as<PolymorphVector>())[2+j+theSystemSize].as<Real>() -1 ;
                }
                else
                {
                    (theH[i+1])[j+1] = (aValueVector[i].as<PolymorphVector>())[2+j+theSystemSize].as<Real>() ;
                }
            }
        }
    }

    GET_METHOD( Polymorph, SSystemMatrix )
    {
        return SSystemMatrix;
    }

    void fire()
    {
        ;
    }

    const boost::multi_array< Real, 2 >& getESSYNSMatrix()
    {
        //get theY
        int anIndex = 0;
        for( VariableReferenceVector::const_iterator i(
                thePositiveVariableReferenceIterator );
             i != theVariableReferenceVector.end() ; ++i )
        {
            if( (*i).getVariable()->getValue() <= 0 )
            {
                THROW_EXCEPTION_INSIDE( ValueError,
                                        asString() + ": the value of "
                                        + (*i).getVariable()->asString()
                                        + " is equal to or less than 0" );
            }
            (theY[anIndex])[0] = gsl_sf_log( (*i).getVariable()->getValue() ) ;
            anIndex++;
        }

        //differentiate first order
        for ( int i( 1 ) ; i < theSystemSize+1 ; i++ )
        {
            Real aGt( 0.0 );
            Real aHt( 0.0 );
            for ( int j( 1 ) ; j < theSystemSize+1 ; j++ )
            {
                aGt += (theG[i])[j] * (theY[j-1])[0];
                aHt += (theH[i])[j] * (theY[j-1])[0];
            }

            Real aAlpha = theAlpha[i] * exp(aGt);
            Real aBate    = theBeta[i] * exp(aHt);

            (theAlphaBuffer[i])[1] = aAlpha;
            (theBetaBuffer[i])[1] = aBate;
            (theY[i-1])[1] =    aAlpha - aBate;
        }

        //differentiate second and/or more order
        for ( int m( 2 ) ; m <= Order ; m++)
        {
            for ( int i( 1 ) ; i < theSystemSize+1 ; i++ )
            {

                (theGBuffer[i])[m-1] = 0;
                (theHBuffer[i])[m-1] = 0;

                for ( int j( 1 ) ; j < theSystemSize+1 ; j++ )
                {
                    const Real aY( (theY[j-1])[m-1] );
                    const Real aG( (theGBuffer[i])[m-1] );
                    const Real aH( (theHBuffer[i])[m-1] );

                    (theGBuffer[i])[m-1] =
                        aG + (theG[i])[j] * aY ;
                    (theHBuffer[i])[m-1] =
                        aH + (theH[i])[j] * aY ;
                }
            }

            for( int i( 1 ) ; i < theSystemSize+1 ; i++ )
            {
                (theAlphaBuffer[i])[m] = 0;
                (theBetaBuffer[i])[m] = 0;

                for( int q( 1 ); 0 > m-q ; q++)
                {
                    (theAlphaBuffer[i])[m] =
                        (theAlphaBuffer[i])[m] +
                        (theFBuffer[m])[q] *
                        (theAlphaBuffer[i])[m-q] *
                        (theGBuffer[i])[m-q] ;
                    (theBetaBuffer[i])[m] =
                        (theBetaBuffer[i])[m]    +
                        (theFBuffer[m])[q] *
                        (theBetaBuffer[i])[m-q] *
                        (theHBuffer[i])[m-q] ;
                }

                (theY[i-1])[m] =
                    (theAlphaBuffer[i])[m] -
                    (theBetaBuffer[i])[m] ;
            }
        }

        return theY;
    }

    virtual Integer getSystemSize() const
    {
        return theSystemSize;
    }

    void initialize()
    {
        Process::initialize();
    }

protected:
    int theSystemSize;
    int Order;

    Polymorph SSystemMatrix;

    // State variables in log space
    boost::multi_array< Real, 2 > theY;

    // S-System vectors
    boost::multi_array< Real, 1 > theAlpha;
    boost::multi_array< Real, 1 > theBeta;

    boost::multi_array< Real, 2 > theG;
    boost::multi_array< Real, 2 > theH;

    // tmp S-System vectors
    boost::multi_array< Real, 2 > theAlphaBuffer;
    boost::multi_array< Real, 2 > theBetaBuffer;
    boost::multi_array< Real, 2 > theGBuffer;
    boost::multi_array< Real, 2 > theHBuffer;
    boost::multi_array< Real, 2 > theFBuffer;

};

LIBECS_DM_INIT( SSystemProcess, Process );
