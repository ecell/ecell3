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
//END_HEADER
/* 	$Id$	 */

#ifndef lint
static char vcid[] = "$Id$";
#endif /* lint */
/* *************************************************************************** */
/* _tableio.c   	--  Fri Feb 20 1998
   A module for reading ASCII tables from files to Python lists.
   
   Important Note: the Python wrapper that calls readTable is
   responsible for making sure that things like '1', '5' and 'e' are
   not valid comment characters.  readTable is quite happy to treat
   '1.5e2' as a comment line.


   Copyright (C) 2000 Michael A. Miller <mmiller@debian.org>

   Time-stamp: <2000-08-09 17:09:14 miller>
   
   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.
   
   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA. 

*/

/* *************************************************************************** */
#include "Python.h"             /* Python header files */

#include <stdio.h>              /* C header files */
#include <string.h>

static PyObject *ErrorObject;   /* locally-raised exception */

#define onError(message) \
    { PyErr_SetString(ErrorObject, message); return NULL; }

/* *************************************************************************** */
/* local logic and data */

#define MAXLINE 32768


/* *************************************************************************** */
/* Exported module method-functions */

static PyObject *
tableio_readTable(PyObject *self, PyObject *args)       /* args: (string) */
{
  char *filename;           /* the name of the file from which to read */
  char *commentchars;       /* the comment characters */
  FILE *infile;             /* the file pointer */
  int status;               /* status flag */

  PyObject *myDict;
  PyObject *myList;

  char line[MAXLINE];      /* line read from the input file */
  char *ct;                /* character tokens for strtok */
  char *string;            /* holder for subtrings */
  char *key;               /* python dictionary key */

  int M;                    /* Number of data lines (rows) in the input file */
  int N;                    /* Numer of data columns in the input file */
  int n;                    /* used to check that N is constant */

  /* Make sure that I've got the arguments that I expect.  In this
     case a string ("s") that holds the filename */
  if (!PyArg_ParseTuple(args, "ss", &filename, &commentchars))
    return NULL;

  /* Check that the file exists and is readable */
  infile = fopen(filename, "r");
  if ( infile == NULL ) 
    {
      onError("Error opening input file");
    }

  /* Determine how many rows and columns are here by skipping past all
     comment lines and parsing the first data line */
  while (fgets(line, sizeof(line), infile) != NULL) 
    {
      /* if ( (strchr(line,'#') == NULL) && (strchr(line,'!') == NULL) ) */
      if ( strpbrk(line,commentchars) == NULL )
	{
	  break;
	}
    }

  /* Create a python list */
  myList = PyList_New(0);

  /* Now go through the line with strtok and see if I can figure out
     how many whitespace delimited fields there are.  
     Notes: 
     - The \n is necessary to take care of trailing whitespace.
     - At each step I check that I can convert the field to a float
       and stick it in a list.  If I can't, I quit.     
  */
  N = -1;
  ct = " \t\n";

  string = strtok( line, ct );
  if ( string != NULL )
    { /* there is at least one field */
      N = 1;
      status = PyList_Append( myList, PyFloat_FromDouble(atof(string)) );	
      if ( status )
	{
	  Py_XDECREF(myList);
	  onError("Error appending to list");
	}
      while (1) 
	{
	  string = strtok( NULL, ct );
	  if ( string != NULL ) 
	    {
	      N = N + 1;
	      status = PyList_Append( myList, PyFloat_FromDouble(atof(string)) );	
	      if ( status ) 
		{
		  Py_XDECREF(myList);
		  onError("Error appending to list");
		}
	    }
	  else 
	    {
	      break;
	    }
	}
    }
  
  /* Go through the rest of the file and fill the list */
  M = 1;
  while (fgets(line, sizeof(line), infile) != NULL) 
    {
      /* If this is not a comment line, treat it as data */
      /* if ( (strchr(line,'#') == NULL) && (strchr(line,'!') == NULL) ) */
      if ( strpbrk(line,commentchars) == NULL )
	{
	  n = -1;
	  
	  string = strtok( line, ct );
	  if ( string != NULL )
	    { /* there is at least one field */
	      n = 1;
	      status = PyList_Append( myList, PyFloat_FromDouble(atof(string)) );	
	      if ( status ) 
		{
		  Py_XDECREF(myList);
		  onError("Error appending to list");
		}
	
	      while ( 1 ) 
		{
		  string = strtok( NULL, ct );
		  if ( string != NULL ) 
		    {
		      n = n + 1;
		      status = PyList_Append( myList, PyFloat_FromDouble(atof(string)) );	
		      if ( status ) 
			{
			  Py_XDECREF(myList);
			  onError("Error appending to list"); 
			}
		    }
		  else 
		    {
		      break;
		    }
		}
	      if ( n != N ) 
		{
		  Py_XDECREF(myList);
		  onError("Error - table has variable number of columns");
		}
	      /* Since I've made it this far, and found data in this
                 line, increment the line count. */
	      M = M + 1;
	    }
	}
    }
  
  /* Close the input file */
  status = fclose(infile);
  if ( status ) {
    Py_XDECREF(myList);
    onError("Error closing input file");
  }
  
  /* If no data was found, report an error and return NULL */
  if ( N == -1 )
    { /* An apparently empty file */
      Py_XDECREF(myList);
      onError("File is apparently empty");
    }

  /* Now, put the data into a dictionary */
  myDict = PyDict_New();
  key = "data";
  status = PyDict_SetItemString(myDict, key, myList);
  if ( status )
    {
      Py_XDECREF(myList);
      Py_XDECREF(myDict);
      onError("Error adding data to dictionary");
    }
  
  key = "rows";
  status = PyDict_SetItemString(myDict, key, Py_BuildValue("i", M));
  if ( status )
    {
      Py_XDECREF(myList);
      Py_XDECREF(myDict);
      onError("Error adding rows to dictionary");
    }

  key = "columns";
  status = PyDict_SetItemString(myDict, key, Py_BuildValue("i", N));
  if ( status )
    {
      Py_XDECREF(myList);
      Py_XDECREF(myDict);
      onError("Error adding columns to dictionary");
    }

  key = "filename";
  status = PyDict_SetItemString(myDict, key, Py_BuildValue("s", filename));
  if ( status ) 
    {
      Py_XDECREF(myList);
      Py_XDECREF(myDict);
      onError("Error adding file name to dictionary");
    }
  
  /* I'm done with my python objects, so DECREF them.  The list can be
     XDECREF'ed, since it isn't used by anyone else. */
  Py_XDECREF(myList);

  /* Finally, return the dictionary to Python */
  return (PyObject *)myDict; 

}

static PyObject *
tableio_writeTable(PyObject *self, PyObject *args)       /* args: (string) */
{
  char *filename;           /* the name of the file from which to read */
  FILE *outfile;            /* the file pointer */
  int status;               /* status flag */

  PyObject *myList;
  PyObject *obj;
  float x;

  int M;                    /* Number of data lines (rows) in the input file */
  int N;                    /* Numer of data columns in the input file */

  int A;                    /* Append mode? -- added by shafi */

  int i, j;
  int item;

  /* Make sure that I've got the arguments that I expect.  In this
     case, a string filename, a list of data, an integer (rows) and
     another integer (columns). */
  if ( ! PyArg_ParseTuple(args, "sOiii", &filename, &myList, &M, &N, &A ) )
    return NULL;

  /* Check that the file exists and is writable */
  if( A != 0 )  /* append mode */
    {
      outfile = fopen(filename, "a");
    }
  else
    {
      outfile = fopen(filename, "w");
    }

  if ( outfile == NULL ) 
    {
      onError("Error opening output file");
    }

  /* make sure M * N = list size */

  /* Write out the data */
  item = -1;
  for (i=0; i<M; i++) 
    {
      for (j=0; j<N; j++) 
	{
	  item = item + 1;
	  obj =  PyList_GetItem(myList,item);
	  if ( !PyArg_Parse( obj, "f", &x) ) {
	    onError("Error reading list");
	  }
	  fprintf(outfile, "%-.15e\t", x);  /* changed -- shafi */
	}
      fprintf(outfile, "\n");
    }

  /* close the file */
  status = fclose(outfile);
  if ( status ) {
    onError("Error closing output file");
  }

  /* Finally, return success */
  return Py_BuildValue("i", 1);
}

/* *************************************************************************** */
/* Method registration table: name-string -> function-pointer */

static struct PyMethodDef tableio_methods[] = {
  {"readTable",  tableio_readTable,   1},
  {"writeTable", tableio_writeTable,  1},
  {NULL,         NULL}
};


/* *************************************************************************** */
/* Initialization function (import-time) */

void
init_tableio()
{
  PyObject *m, *d;

  /* create the module and add the functions */
  m = Py_InitModule("_tableio", tableio_methods);        /* registration hook */
  
  /* add symbolic constants to the module */
  d = PyModule_GetDict(m);
  ErrorObject = Py_BuildValue("s", "tableio.error");   /* export exception */
  PyDict_SetItemString(d, "error", ErrorObject);       /* add more if need */
  
  /* check for errors */
  if (PyErr_Occurred())
    Py_FatalError("can't initialize module tableio");
}

/* End of tableio.c */
