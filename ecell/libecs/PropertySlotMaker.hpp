#ifndef __PROPERTYSLOTMAKER_HPP
#define __PROPERTYSLOTMAKER_HPP

#include "libecs.hpp"

#include "Exceptions.hpp"
#include "PropertySlot.hpp"
#include "PropertiedClass.hpp"



namespace libecs
{


  /** @addtogroup property
   *@{
   */

  /** @file */

  /**


     @internal
  */


  template
  <
    class T
    >
  class PropertySlotMaker
  {
    
  public:
    
    template<typename SlotType>
    static PropertySlot<T>*
    createPropertySlot( Type2Type<SlotType>,
			typename ConcretePropertySlot<T,SlotType>::SetMethodPtr
			aSetMethodPtr,
			typename ConcretePropertySlot<T,SlotType>::GetMethodPtr
			aGetMethodPtr )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  aSetMethodPtr = &PropertiedClass::nullSet;
	}
      
      if( aGetMethodPtr == NULLPTR )
	{
	  aGetMethodPtr = &PropertiedClass::nullGet<SlotType>;
	}
      

      PropertySlot<T>* 
	aPropertySlotPtr( new ConcretePropertySlot<T,SlotType>
			  ( aSetMethodPtr,
			    aGetMethodPtr ) );

      return aPropertySlotPtr;
    }

  };


  /*@}*/

}

#endif /* __PROPERTYSLOTMAKER_HPP */
