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

#include <gsl/gsl_sf.h>
#include <boost/multi_array.hpp>

#include <libecs/libecs.hpp>
#include <libecs/System.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/Variable.hpp>
#include <libecs/Process.hpp>
#include <libecs/Util.hpp>
#include <libecs/PropertyInterface.hpp>

#include <libecs/ContinuousProcess.hpp>

#include "ESSYNSProcessInterface.hpp"

USE_LIBECS;

// Although this class is not exactly a continuous process
// (this doesn't set the activity),  the method below is defined
// just to suppless the warning in ESSYNSStepper's
// superclass, AdaptiveDifferentialStepper.   some more thoughts on
// Process ontology will be necessary.

LIBECS_DM_CLASS_EXTRA_1( GMAProcess, ContinuousProcess, ESSYNSProcessInterface )
{
public:

    LIBECS_DM_OBJECT( GMAProcess, Process )
    {
        INHERIT_PROPERTIES( ContinuousProcess );
     
        PROPERTYSLOT_SET_GET( Integer, Order );
        PROPERTYSLOT_SET_GET( Polymorph, GMASystemMatrix );
    }
    
    GMAProcess()
        : theSystemSize(0), Order(3)
    {
        ; // do nothing.
    }

    virtual ~GMAProcess()
    {
        ;
    }

    SIMPLE_GET_METHOD( Integer, Order );

    SET_METHOD( Integer, Order )
    { 
        Order = value;
        
        // init Substance Vector
        theY.resize( boost::extents[ theSystemSize + 1 ][ Order + 1 ] );

        // init GMA Vector & Matrix
        theAlpha.resize( boost::extents[ theLawSize ][ theLawSize ] );
        theG.resize( boost::extents[ theLawSize ][ theLawSize ][ theLawSize ] );

        // init S-System tmp Vector & Matrix
        theAlphaBuffer.resize( boost::extents[ theLawSize ][ theLawSize ][ Order + 1 ] );
        theGBuffer.resize( boost::extents[ theLawSize ][ theLawSize ][ Order + 1 ] );

        theFBuffer.resize( boost::extents[ Order + 1 ][ Order ] );
    }
    
    SET_METHOD( Polymorph, GMASystemMatrix )
    {
        GMASystemMatrix = value;
        PolymorphVector aValueVector( value.as<PolymorphVector>() );
        theSystemSize = aValueVector.size();
        theLawSize = theSystemSize + 1;

        // init Substance Vector
        theY.resize( boost::extents[ theLawSize ][ Order + 1 ] );

        // init GMA-System Vector & Matrix
        theAlpha.resize( boost::extents[ theLawSize ][ theLawSize ] );
        theG.resize( boost::extents[ theLawSize ][ theLawSize ][ theLawSize ] );

        // init GMA-System tmp Vector & Matrix
        theAlphaBuffer.resize( boost::extents[ theLawSize ][ theLawSize ][ Order + 1] );
        theGBuffer.resize( boost::extents[ theLawSize ][ theLawSize ][ Order + 1 ] );

        theFBuffer.resize( boost::extents[ Order + 1 ][ Order ] );

        // init Factorial matrix
        for(Integer m( 2 ) ; m < Order+1 ; m++) {
            for(Integer q( 1 ); q < m ; q++) {
                const Real aFact( 1 / gsl_sf_fact(q-1) * gsl_sf_fact(m-q-1) * m * (m-1) );
                (theFBuffer[m])[q] = aFact;      
            }
        }

        // set Alpha, Beta, G, H 
        for ( std::size_t i( 0 ); i < theSystemSize ; i++ )
        {
            for ( std::size_t j( 0 ); j < theSystemSize; j++ )
            {
                theAlpha[i+1][j+1] = (aValueVector[i].as<PolymorphVector>())[j].as<Real>() ;          
                for ( std::size_t k( 0 ); k < theSystemSize; k++ )
                {
                    if ( i == k )
                    {
                        
                        (theG[i+1])[j+1][k+1] = ((aValueVector[i].as<PolymorphVector>())[theSystemSize + j ].as<PolymorphVector>())[k].as<Real>() -1;  
                    }
                    else
                    {
                        
                        (theG[i+1])[j+1][k+1] = ((aValueVector[i].as<PolymorphVector>())[theSystemSize + j ].as<PolymorphVector>())[k].as<Real>();
                    }              
                }
                
            }
        }
    }

    GET_METHOD( Polymorph, GMASystemMatrix )
    {
        return GMASystemMatrix;
    }
    
    void fire()
    {
        ;
    }

    virtual const boost::multi_array< Real, 2 >& getESSYNSMatrix()
    {
        // get theY
        Integer anIndex( 0 );
        
        for ( VariableReferenceVectorConstIterator
               i ( thePositiveVariableReferenceIterator );
             i != theVariableReferenceVector.end(); ++i )
        {
            if ( (*i).getVariable()->getValue() <= 0 )
            {
                THROW_EXCEPTION_INSIDE( ValueError, 
                                        asString() +
                                        ": the value of Variable ["
                                        + (*i).getVariable()->asString()
                                        + " went under 0" );
            }

            (theY[anIndex])[0] = gsl_sf_log( (*i).getVariable()->getValue() ) ;

            anIndex++;
        }

        // differentiate first order
        Real aGt( 0.0 );  
        Real aAlpha( 0.0 );         
        for ( std::size_t i( 1 ); i < theLawSize; i++ )
        {
            (theY[i-1])[1] = 0;//reset theY
            for ( std::size_t j( 1 ) ; j < theLawSize; j++ )
            {
                aGt = 0.0;//reset aGt
                for ( std::size_t k( 1 ) ; k < theLawSize ; k++ )
                {
                    aGt += ( (theG[i])[j][k] * (theY[k-1])[0] );
                }
                
                aAlpha = 0.0;
                aAlpha = ( theAlpha[i][j] * exp( aGt ) );         
                
                (theAlphaBuffer[i])[j][1] = aAlpha;
                (theY[i-1])[1] +=  aAlpha;
            }
        }

        // differentiate second and/or more order
        for ( Integer m( 2 ); m <= Order; m++ )
        {
            for ( std::size_t i( 1 ) ; i < theLawSize; i++ )
            {
                for ( std::size_t j( 1 ); j < theLawSize; j++ )
                {
                    (theGBuffer[i])[j][m] = 0; //reset GBuffer        
                    (theAlphaBuffer[i])[j][m] = 0; //reset ABuffer
                    
                    for( std::size_t k( 1 ); k < theLawSize; k++ )
                    {
                        (theGBuffer[i])[j][m - 1] += 
                          ( (theG[i])[j][k] * (theY[k - 1])[m - 1] ); 
                    }
                    for( Integer q( 1 );  q <= m - 1; q++)
                    {
                        (theAlphaBuffer[i])[j][m] +=  
                          ( (theFBuffer[m])[q]*
                            (theAlphaBuffer[i])[j][m-q]* 
                            (theGBuffer[i])[j][q] );
                    }
                   
                    (theY[i-1])[m] = (theAlphaBuffer[i])[j][m] / (m-1);
                }
            } 
        }
        
        return theY;
    }

    virtual Integer getSystemSize() const
    {
        return static_cast< Integer >( theSystemSize );
    }

    void initialize()
    {
        Process::initialize();
    }  
    
protected:

    Integer Order;
    std::size_t theSystemSize;
    std::size_t theLawSize;

    Polymorph GMASystemMatrix;
 
    // State variables in log space
    boost::multi_array< Real, 2 > theY;

    // GMA-System vectors
    boost::multi_array< Real, 2 > theAlpha;

    boost::multi_array< Real, 3 > theG;

    // tmp GMA-System vectors
    boost::multi_array< Real, 3 > theAlphaBuffer;
    boost::multi_array< Real, 3 > theGBuffer;
    boost::multi_array< Real, 2 > theFBuffer;
};

LIBECS_DM_INIT( GMAProcess, Process );
