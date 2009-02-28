//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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

#include <gsl/gsl_sf.h>
#include <boost/multi_array.hpp>

#include <libecs/Variable.hpp>
#include <libecs/Process.hpp>
#include <libecs/PropertyInterface.hpp>
#include <libecs/AdaptiveDifferentialStepper.hpp>

#include "ESSYNSProcessInterface.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ESSYNSStepper, AdaptiveDifferentialStepper )
{
public:
    LIBECS_DM_OBJECT( ESSYNSStepper, Stepper )
    {
        INHERIT_PROPERTIES( AdaptiveDifferentialStepper );
        PROPERTYSLOT_SET_GET( Integer, TaylorOrder );
    }

    GET_METHOD( Integer, TaylorOrder )
    {
        return theTaylorOrder;
    }

    SET_METHOD( Integer, TaylorOrder )
    {
        theTaylorOrder = value;
    }

    ESSYNSStepper()
        : theESSYNSProcessPtr( NULLPTR, NULLPTR ), theTaylorOrder( 1 )
    {
        ; 
    }
	        
    virtual ~ESSYNSStepper()
    {
        ;
    }

    virtual void initialize()
    {
        AdaptiveDifferentialStepper::initialize();
     
        // initialize()

        if( theProcessVector.size() == 1 )
        {
            theESSYNSProcessPtr = std::make_pair(
                SafeDynamicCast< ESSYNSProcessInterface* >( theProcessVector[ 0 ] ),
                theProcessVector[ 0 ] );
            theSystemSize = theESSYNSProcessPtr.first->getSystemSize();
        }
        else
        {
            THROW_EXCEPTION( InitializationFailed, 
                   "Error:in ESYYNSStepper::initialize() " );
        }

        theTaylorOrder = getOrder();

        theESSYNSMatrix.resize( boost::extents[ theSystemSize + 1 ][ theTaylorOrder + 1 ] );

        theIndexVector.resize( theSystemSize );
        VariableReferenceVectorCref aVariableReferenceVectorCref(
            theESSYNSProcessPtr.second->getVariableReferenceVector() );

        for ( VariableReferenceVector::size_type c(
                theESSYNSProcessPtr.second->getPositiveVariableReferenceOffset() );
              c < theSystemSize; ++c )
        {
            VariableReferenceCref aVariableReferenceCref( aVariableReferenceVectorCref[ c ] );
            const VariablePtr aVariablePtr( aVariableReferenceCref.getVariable() );

            theIndexVector[ c ] = getVariableIndex( aVariablePtr );
        }
    }

    virtual bool calculate()
    {
        const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

        Real aCurrentTime( getCurrentTime() );
        Real aStepInterval( getStepInterval() );

        // write step() function
        theESSYNSMatrix = theESSYNSProcessPtr.first->getESSYNSMatrix();

        //integrate
        for( int i( 1 ); i < theSystemSize + 1; i++ )
        {
            Real aY( 0.0 ); //reset aY 
            for( int m( 1 ); m <= theTaylorOrder; m++ )
            {
                aY += ((theESSYNSMatrix[i-1])[m] *
                gsl_sf_pow_int( aStepInterval, m ) / gsl_sf_fact( m ));
            }
            (theESSYNSMatrix[i-1])[0] += aY;
            //std::cout<< (theESSYNSMatrix[i-1])[0] <<std::endl;
        }
        
        //set value
        for( int c( 0 ); c < aSize; ++c )
        {
            const VariableVector::size_type anIndex( theIndexVector[ c ] );
            VariablePtr const aVariable( theVariableVector[ anIndex ] );
        
            const Real aVelocity( ( exp( (theESSYNSMatrix[c])[0] ) - ( aVariable->getValue() ) ) / aStepInterval );
                 
            theTaylorSeries[ 0 ][ anIndex ] = aVelocity;
        }

        return true;
    }
        
    virtual GET_METHOD( Integer, Order )
    {
        return theTaylorOrder;
    }

    virtual GET_METHOD( Integer, Stage ) {
        return 1;
    }

protected:

    Integer theSystemSize;
    Integer theTaylorOrder;
    std::pair< ESSYNSProcessInterface*, Process* > theESSYNSProcessPtr;
    boost::multi_array< Real, 2 > theESSYNSMatrix;
    std::vector<VariableVector::size_type> theIndexVector;
};

LIBECS_DM_INIT( ESSYNSStepper, Stepper );
