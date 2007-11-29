//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
// Contact information:
//   Nathan Addy, Research Associate     Voice: 510-981-8748
//   The Molecular Sciences Institute    Email: addy@molsci.org  
//   2168 Shattuck Ave.                  
//   Berkeley, CA 94704
//
//END_HEADER

#ifndef __APOPTOSISPROCESS_HPP
#define __APOPTOSISPROCESS_HPP

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>

USE_LIBECS;

/**************************************************************

       ApoptosisProcess

**************************************************************/

LIBECS_DM_CLASS( ApoptosisProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( ApoptosisProcess, Process)
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( String, Type );
cd      PROPERTYSLOT_SET_GET( String, GreaterOrLessThan );
      PROPERTYSLOT_SET_GET( Real, Expression );
    }



  ApoptosisProcess();
  ~ApoptosisProcess();

  virtual void initialize();
  virtual void fire();

  SET_METHOD( String, Type);
  SET_METHOD( String, GreaterOrLessThan );
  SET_METHOD( Real, Expression );

  GET_METHOD( String, Type);
  GET_METHOD( String, GreaterOrLessThan );
  GET_METHOD( Real, Expression );


 private:

  void destroyCell();

  // Corresponds to "Type".
  bool expressionChecksConcentration;
  Real apoptosisThreshold;
  bool apoptosisBelowThreshold;

  SystemPtr cellToBeKilled;

};

#endif
