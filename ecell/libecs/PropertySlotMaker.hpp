#ifndef __PROPERTYSLOTMAKER_HPP
#define __PROPERTYSLOTMAKER_HPP

#include "libecs.hpp"

#include "Exceptions.hpp"
#include "PropertySlot.hpp"
#include "PropertyInterface.hpp"



namespace libecs
{


  /** @addtogroup property
   *@{
   */

  /** @file */

  /**


     @internal
  */

  class PropertySlotMaker
  {
    
  public:
    
    template<class T, typename SlotType>
    static PropertySlotPtr 
    createPropertySlot( StringCref aName,
			T& anObject,
			Type2Type<SlotType>,
			typename ConcretePropertySlot<T,SlotType>::SetMethodPtr
			aSetMethodPtr,
			typename ConcretePropertySlot<T,SlotType>::GetMethodPtr
			aGetMethodPtr,
			typename ConcretePropertySlot<T,SlotType>::SetMethodPtr
			aSyncMethodPtr = NULLPTR )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  aSetMethodPtr = &PropertyInterface::nullSet;
	}
      
      if( aGetMethodPtr == NULLPTR )
	{
	  aGetMethodPtr = &PropertyInterface::nullGet<SlotType>;
	}
      

      PropertySlotPtr aPropertySlotPtr( NULLPTR );


      // if sync method is not given, create without it.
      if( aSyncMethodPtr == NULLPTR )
	{
	  aPropertySlotPtr = 
	    new ConcretePropertySlot<T,SlotType>( aName,
						  anObject,
						  aSetMethodPtr,
						  aGetMethodPtr );
	}
      else // create with a sync method.
	{
	  aPropertySlotPtr = 
	    new ConcretePropertySlot<T,SlotType>( aName,
						  anObject,
						  aSetMethodPtr,
						  aGetMethodPtr,
						  aSyncMethodPtr );
	}

      return aPropertySlotPtr;
    }

  };




  /*@}*/

}

#endif /* __PROPERTYSLOTMAKER_HPP */
