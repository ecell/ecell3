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

#include "RCPtr.hpp"



// empty




#ifdef DEBUG_RCPTR

using namespace libecs;

/*
 * Example of how to use the AutoRelease Library
 *
 * Output of this program:
 *
 * Entering main-scope
 * (Creating Hello object with message 'Hello 1')
 * (Creating Hello object with message 'Hello 2')
 * (Creating Hello object with message 'Hello 3')
 * (Creating Hello object with message 'Hello 4')
 * (Deleting Hello object with message 'Hello 4')
 * Hello 3
 * (Deleting Hello object with message 'Hello 1')
 * Hello 3
 * Leaving main-scope. a, b, c, and d goes out-of-scope
 * (Deleting Hello object with message 'Hello 2')
 * (Deleting Hello object with message 'Hello 3')
 */


class Hello {
public:
    Hello(char* msg);
    virtual ~Hello();
    
    void print();

private:
    char* message;
};


Hello::Hello(char* msg) {
    message = msg;
    cout << "(Creating Hello object with message '" << message << "')" << endl;
}


Hello::~Hello() {
    cout << "(Deleting Hello object with message '" << message << "')" << endl;
}


void Hello::print() {
    cout << message << endl;
}

DECLARE_TYPE( RCPtr<Hello>, HelloRCPtr );


// This is how you declare a function to take a reference 
// counted pointer as parameter as well as returning one. 
//HelloRCPtr print( HelloRCPtr h ) {
RCPtr<Hello> print( HelloRCPtr h ) {
    h->print();
    return h;
}


int main() {
    cout << "Entering main-scope" << endl; 

    // Create objects
    HelloRCPtr a( new Hello("Hello 1") );
    HelloRCPtr b( new Hello("Hello 2") );
    HelloRCPtr c( new Hello("Hello 3") );
    HelloRCPtr d;
    d = new Hello("Hello 4");
    
    d = c;  // Releasing reference to "Hello 4", which will be deleted.

    a = print(d);  // Releasing reference to "Hello 1", which will be deleted.
    d->print();    // Print "Hello 3".

    cout << "Leaving main-scope. a, b, c, and d goes out-of-scope" << endl; 

    // No need to make delete here, all objects will be deleted automatically.
}

#endif /* DEBUG_RCPTR */
