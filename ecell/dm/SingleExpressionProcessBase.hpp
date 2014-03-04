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
// authors:
//   Koichi Takahashi
//   Tatsuya Ishida
//   Yasuhiro Naito
//
// E-Cell Project.
//

#ifndef __SINGLEEXPRESSIONPROCESSBASE_HPP
#define __SINGLEEXPRESSIONPROCESSBASE_HPP

#include "ExpressionProcessBase.hpp"

template< typename Tmixin_ >
class SingleExpressionProcessBase : public ExpressionProcessBase< Tmixin_ >
{
protected:

private:
    friend class PropertyAccess;
    friend class EntityResolver;
    friend class VariableReferenceResolver;
    friend class ErrorReporter;

public:

    LIBECS_DM_OBJECT_MIXIN( SingleExpressionProcessBase, Tmixin_ )
    {
        PROPERTYSLOT_SET_GET( libecs::String, Expression );
    }


    SingleExpressionProcessBase()
        : theCompiledCode( 0 ), theRecompileFlag( true )
    {
        // ; do nothing
    }

    virtual ~SingleExpressionProcessBase()
    {
        delete theCompiledCode;
    }

    SET_METHOD( libecs::String, Expression )
    {
        theExpression = value;
        theRecompileFlag = true;
    }

    GET_METHOD( libecs::String, Expression )
    {
        return theExpression;
    }

    void initialize()
    {
        if ( theRecompileFlag )
        {
            try
            {
                this->compileExpression( theExpression, theCompiledCode );
            }
            catch ( libecs::Exception const& e )
            {
                throw libecs::InitializationFailed( e, static_cast< Tmixin_ * >( this ) );
            }
            theRecompileFlag = false;
        }
    }

protected:

protected:
    libecs::String    theExpression;

    const libecs::scripting::Code* theCompiledCode;
    libecs::scripting::VirtualMachine theVirtualMachine;

    bool theRecompileFlag;
};

#endif /* __SINGLEEXPRESSIONPROCESSBASE_HPP */

