//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

#ifndef ___FQPI_H___
#define ___FQPI_H___
#include <string>

#include "libecs.hpp"
#include "Exceptions.hpp"
#include "PrimitiveType.hpp"

namespace libecs
{



  /** 
      SystemPath 
  */
  class SystemPath {

  public:

    SystemPath( StringCref rqsn = "" );
    SystemPath( SystemPathCref systempath );
    virtual ~SystemPath() {}

    StringCref getSystemPathString() const { return theSystemPath; }
    virtual const String getString() const { return getSystemPathString(); }
    virtual operator String() const { return getSystemPathString(); }

    /**
       Extract the first system name.
       @return name of the first system
    */
    const String first() const;

    /**
       Extract the last system name.

       @return name of the last system in given systempath.
    */
    const String last() const;

    /**
       @return SystemPath without the first system name.
    */
    SystemPath next() const;

  protected:

    /**
       Standardize a SystemPath. 
       Reduce '..'s and remove extra white spaces.

       @return reference to the systempath
    */
    void standardize();

    SystemPath() {}

  private:

    SystemPathRef operator=( SystemPathCref rhs );

  public:

    static const char DELIMITER = '/';

  private:

    const String theSystemPath;

  };

  /**
     FQID(Fully Qualified entity ID)

     The FQID is a identifier (ID) of Entity objects of certain PrimitiveType.
     Given a PrimitiveType, one can identify unique Entity in a
     cell model with a SystemPath and an id.  

     @see SystemPath
  */
  class FQID : public SystemPath
  {

  public:

    FQID( StringCref systemname, StringCref id );
    FQID( StringCref fqid );
    FQID( FQIDCref fqid );
    virtual ~FQID() {}

    const String getFqidString() const;
    virtual const String getString() const { return getFqidString(); }
    StringCref getIdString() const { return theId; }
    virtual operator String() const { return getFqidString(); }

    static const String IdOf( StringCref fqen );
    static const String SystemPathOf( StringCref fqen );

  private:

    FQIDRef operator=( FQIDCref rhs );

  private:

    const String theId;

  };

  /**
     FQPI (Fully Qualified Primitive Id).

     One can identify an unique Entiy in a cell model with a FQPI.
     The FQPI consists of FQID and PrimitiveType.

     @see FQID, PrimitiveType
  */
  class FQPI : public FQID
  {

  public:

    static const String  fqidOf( StringCref fqpi );

    FQPI( const PrimitiveType type, FQIDCref fqid );
    FQPI( StringCref fqpistring );
    FQPI( FQPICref fqpi );
    virtual ~FQPI() {}
  
    const String getFqpiString() const;
    const PrimitiveType getPrimitiveType() const { return thePrimitiveType; }

    virtual const String getString() const { return getFqpiString(); }
    virtual operator const String() const { return getFqpiString(); }

  private:

    FQPIRef operator=( FQPICref rhs );

  private:

    PrimitiveType thePrimitiveType;

  };


} // namespace libecs

#endif /*  ___FQPI_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
