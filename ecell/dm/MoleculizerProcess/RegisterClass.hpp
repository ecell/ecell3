#ifndef __REGISTERCLASS_HPP
#define __REGISTERCLASS_HPP

#include <string>
#include <vector>

class RegisterClass
{
public:
  RegisterClass()
  {}

  void 
  addString(std::string i)
  {
    theVect.push_back(i);
  }

  void 
  clearAll()
  {
    theVect.clear();
  }

  std::vector<std::string> theVect;
};

#endif
