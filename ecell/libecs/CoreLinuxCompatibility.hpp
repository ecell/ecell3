//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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


//
//  This file is a set of cut-and-pastes from libcorelinux.
//


/*
  CoreLinux++ 
  Copyright (C) 1999 CoreLinux Consortium
  
   The CoreLinux++ Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The CoreLinux++ Library Library is distributed in the hope that it will 
   be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If not,
   write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
   Boston, MA 02111-1307, USA.  
*/   



namespace corelinux
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  // from Common.hpp

#define IN_COMMON_HPP


  /**
     IGNORE_RETURN is an indicator that the return
     value for a function is ignored.
     i.e   IGNORE_RETURN getSomething( ... );
     Eliminates a lint warning.
  */

#define IGNORE_RETURN (void)

  /**
     Declare a new type and its pointer,
     const pointer, reference, and const reference types. For example
     DECLARE_TYPE( Dword, VeryLongTime );
     @param mydecl The base type
     @param mytype The new type
  */

#define DECLARE_TYPE( mydecl, mytype )  \
typedef mydecl         mytype;         \
typedef mytype *       mytype ## Ptr;  \
typedef const mytype * mytype ## Cptr; \
typedef mytype &       mytype ## Ref;  \
typedef const mytype & mytype ## Cref;

  /**
     Declare class , class pointer , 
     const pointer, class reference 
     and const class reference types for classes. For example
     DECLARE_CLASS( Exception );
     @param tag The class being declared
  */

#define DECLARE_CLASS( tag )            \
   class   tag;                        \
   typedef tag *       tag ## Ptr;     \
   typedef const tag * tag ## Cptr;    \
   typedef tag &       tag ## Ref;     \
   typedef const tag & tag ## Cref;


  // from Types.hpp


  // *******************************************
  // Floating Point.
  // *******************************************

  DECLARE_TYPE( double, Real );
   
  // *******************************************
  // Define the void pointer type.
  // *******************************************
   
  typedef void * VoidPtr;

  // *******************************************
  // Define the NULLPTR
  // *******************************************

#define  NULLPTR  0
   



  // from List.hpp

  /**
     STL list template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a list.
     @param name The name you want to give the collection
     @param type The type object the collection manages
  */
#define CORELINUX_LIST( type, name )                            \
      DECLARE_TYPE(std::list<type>,name);                       \
      typedef name::iterator name ## Iterator;                  \
      typedef name::iterator& name ## IteratorRef;              \
      typedef name::iterator* name ## IteratorPtr;              \
      typedef name::const_iterator name ## ConstIterator;       \
      typedef name::const_iterator& name ## ConstIteratorRef;   \
      typedef name::const_iterator* name ## ConstIteratorPtr;   \
      typedef name::reverse_iterator name ## Riterator;         \
      typedef name::reverse_iterator& name ## RiteratorRef;     \
      typedef name::reverse_iterator* name ## RiteratorPtr


  // from Vector.hpp

  /**
     STL vector template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a vector.
     @param name The name you want to give the collection
     @param type The type for the vector
  */
#define CORELINUX_VECTOR( type, name )                            \
   DECLARE_TYPE(std::vector<type>,name);                       \
   typedef name::iterator name ## Iterator;                    \
   typedef name::iterator& name ## IteratorRef;                \
   typedef name::iterator* name ## IteratorPtr;                \
   typedef name::const_iterator name ## ConstIterator;         \
   typedef name::const_iterator& name ## ConstIteratorRef;     \
   typedef name::const_iterator* name ## ConstIteratorPtr;     \
   typedef name::reverse_iterator name ## Riterator;           \
   typedef name::reverse_iterator& name ## RiteratorRef;       \
   typedef name::reverse_iterator* name ## RiteratorPtr


  // from Set.hpp

  /**
     STL set template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a set.
     @param name The name you want to give the collection
     @param key The object that represents the set key
     @param comp The comparator functor
  */
#define CORELINUX_SET(key,comp,name)                                       \
      typedef set<key, comp > name;                                           \
      typedef name *       name ## Ptr;                                       \
      typedef const name * name ## Cptr;                                      \
      typedef name &       name ## Ref;                                       \
      typedef const name & name ## Cref;                                      \
      typedef name::iterator name ## Iterator;                                \
      typedef name::iterator& name ## IteratorRef;                            \
      typedef name::iterator* name ## IteratorPtr;                            \
      typedef name::const_iterator name ## ConstIterator;                     \
      typedef name::const_iterator& name ## ConstIteratorRef;                 \
      typedef name::const_iterator* name ## ConstIteratorPtr;                 \
      typedef name::reverse_iterator name ## Riterator;                       \
      typedef name::reverse_iterator& name ## RiteratorRef;                   \
      typedef name::reverse_iterator* name ## RiteratorPtr
   
  /**
     STL multiset template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a multiset.
     @param name The name you want to give the collection
     @param key The object that represents the mutliset key
     @param comp The comparator functor
  */
#define CORELINUX_MULTISET(key,comp,name)                                  \
      typedef multiset<key, comp > name;                                      \
      typedef name *       name ## Ptr;                                       \
      typedef const name * name ## Cptr;                                      \
      typedef name &       name ## Ref;                                       \
      typedef const name & name ## Cref;                                      \
      typedef name::iterator name ## Iterator;                                \
      typedef name::iterator& name ## IteratorRef;                            \
      typedef name::iterator* name ## IteratorPtr;                            \
      typedef name::const_iterator name ## ConstIterator;                     \
      typedef name::const_iterator& name ## ConstIteratorRef;                 \
      typedef name::const_iterator* name ## ConstIteratorPtr;                 \
      typedef name::reverse_iterator name ## Riterator;                       \
      typedef name::reverse_iterator& name ## RiteratorRef;                   \
      typedef name::reverse_iterator* name ## RiteratorPtr


  // from Map.hpp


  /**
     STL map template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a map.
     @param name The name you want to give the collection
     @param key The object that represents the map key
     @param value The object that the key is associated to
     @param comp The comparator functor
  */
#define CORELINUX_MAP(key,value,comp,name)                             \
      typedef std::map<key,value,comp > name;                      \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr
   
  /**
     STL multimap template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a multimap.
     @param name The name you want to give the collection
     @param key The object that represents the map key
     @param value The object that the key is associated to
     @param comp The comparator functor
  */

#define CORELINUX_MULTIMAP(key,value,comp,name)                 \
      typedef std::multimap<key,value,comp > name;                 \
      typedef name *       name ## Ptr;                            \
      typedef const name * name ## Cptr;                           \
      typedef name &       name ## Ref;                            \
      typedef const name & name ## Cref;                           \
      typedef name::iterator name ## Iterator;                     \
      typedef name::iterator& name ## IteratorRef;                 \
      typedef name::iterator* name ## IteratorPtr;                 \
      typedef name::const_iterator name ## ConstIterator;          \
      typedef name::const_iterator& name ## ConstIteratorRef;      \
      typedef name::const_iterator* name ## ConstIteratorPtr;      \
      typedef name::reverse_iterator name ## Riterator;            \
      typedef name::reverse_iterator& name ## RiteratorRef;        \
      typedef name::reverse_iterator* name ## RiteratorPtr


  // from Queue.hpp

  /**
     STL queue template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a queue.
     @param name The name you want to give the collection
     @param type The type to be queued
  */
#define CORELINUX_QUEUE( type, name )                          \
      DECLARE_TYPE(std::deque<type>,name);                     \
      typedef name::iterator name ## Iterator;                 \
      typedef name::iterator& name ## IteratorRef;             \
      typedef name::iterator* name ## IteratorPtr;             \
      typedef name::const_iterator name ## ConstIterator;      \
      typedef name::const_iterator& name ## ConstIteratorRef;  \
      typedef name::const_iterator* name ## ConstIteratorPtr;  \
      typedef name::reverse_iterator name ## Riterator;        \
      typedef name::reverse_iterator& name ## RiteratorRef;    \
      typedef name::reverse_iterator* name ## RiteratorPtr


  // from Stack.hpp

  /**
     STL stack template. This macro generates all
     the type references and pointers for the collection and
     respective iterators for a stack.
     @param name The name you want to give the collection
     @param type The type to be stacked
  */
#define CORELINUX_STACK( type, name )                                 \
      DECLARE_TYPE(stack<type>,name)                                   

  /** @} */ //end of libecs_module 

} // namespace corelinux
