//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef __RULESRMREACTOR_HPP
#define __RULESRMREACTOR_HPP

#include "SRMReactor.hpp"

namespace libecs
{

  DECLARE_CLASS( RuleSRMReactor );

  class RuleSRMReactor
    :
    public SRMReactor
  {

  public:

    class IsRuleReactor
      : 
      public std::unary_function<SRMReactorPtr,bool>
    {
    public:
      result_type operator()( const argument_type aReactorPtr ) const
      {
	if( dynamic_cast<RuleSRMReactorPtr> ( aReactorPtr ) != NULLPTR ) 
	  {
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }
    };



    RuleSRMReactor()
    {
      ; // do nothing
    }

    virtual ~RuleSRMReactor()
    {
      ; // do nothing
    }


  };

} // namespace libecs


#endif /* __RULESRMREACTOR_HPP */
