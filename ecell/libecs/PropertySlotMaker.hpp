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
    createPropertySlot( T& anObject,
			Type2Type<SlotType>,
			typename ConcretePropertySlot<T,SlotType>::SetMethodPtr
			aSetMethodPtr,
			typename ConcretePropertySlot<T,SlotType>::GetMethodPtr
			aGetMethodPtr )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  aSetMethodPtr = &PropertyInterface::nullSet;
	}
      
      if( aGetMethodPtr == NULLPTR )
	{
	  aGetMethodPtr = &PropertyInterface::nullGet<SlotType>;
	}
      

      PropertySlotPtr aPropertySlotPtr( 
	new ConcretePropertySlot<T,SlotType>( anObject,
					      aSetMethodPtr,
					      aGetMethodPtr ) );

      return aPropertySlotPtr;
    }

  };


  /*@}*/

}

#endif /* __PROPERTYSLOTMAKER_HPP */
