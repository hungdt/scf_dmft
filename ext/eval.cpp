#include <alps/scheduler.h>
#include <alps/alea/detailedbinning.h>

#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <numpy/arrayobject.h>
#include <string>

#include <iostream>

PyObject * get_raw_data(boost::python::str &path, const int &measure, PyObject *obs_str_list)
{
    import_array1(NULL);
    char *s = boost::python::extract<char *>(path);
    std::string str_parms = std::string(s) + ".task1.out.xml";
    alps::scheduler::SimpleMCFactory<alps::scheduler::DummyMCRun> factory;
    alps::scheduler::init(factory);
    boost::filesystem::path p(str_parms.c_str(), boost::filesystem::native);
    alps::ProcessList nowhere;
    alps::scheduler::MCSimulation sim(nowhere,p);

    alps::RealVectorObsevaluator G=sim.get_measurements()["Greens"];
    int N = int(sim.get_parameters()["N"]),
        FLAVORS = int(sim.get_parameters()["SPINS"])*int(sim.get_parameters()["SITES"]);
    alps::RealVectorObsevaluator n=sim.get_measurements()["n"];
    
    npy_intp dim[2] = {N+1, FLAVORS};
    PyArrayObject *Gtau = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_DOUBLE),
                  *Gerr = (PyArrayObject *) PyArray_SimpleNew(2, dim, NPY_DOUBLE);
    Py_INCREF(Gtau); Py_INCREF(Gerr);
    double *buf = (double *)PyArray_DATA(Gtau),
           *buf_err = (double *)PyArray_DATA(Gerr);

    for (int j=0; j<FLAVORS; j++) {
        buf[j] =  -(1-n.mean()[j]);
        buf_err[j] = n.error()[j];
    }
    for (int i=1; i<N; i++) 
        for (int j=0; j<FLAVORS; j++) {
            buf[j + i*FLAVORS] = -(G.mean()[j*(N+1)+i]);
            buf_err[j + i*FLAVORS] = G.error()[j*(N+1)+i];
        }
    for (int j=0; j<FLAVORS; j++) {
        buf[j + N*FLAVORS] =  -(n.mean()[j]);
        buf_err[j + N*FLAVORS] =  n.error()[j];
    }

    // measure other observables
    PyObject *obs = Py_None;
    Py_INCREF(Py_None);
    if (measure > 0) {
        int Nlist = PyList_Size(obs_str_list);
        Py_DECREF(Py_None);
        obs = PyDict_New();
        for (int i = 0; i < Nlist; ++i) {
            alps::RealVectorObsevaluator o = sim.get_measurements()[PyString_AsString(PyList_GetItem(obs_str_list, i))];
            npy_intp o_size = o.mean().size();
            PyArrayObject *tmp = (PyArrayObject *) PyArray_SimpleNew(1, &o_size, NPY_DOUBLE);
            double *buffer = (double *)PyArray_DATA(tmp);
            for (int j = 0; j < o_size; ++j) 
                buffer[j] = o.mean()[j];
            PyDict_SetItem(obs, PyList_GetItem(obs_str_list, i), PyArray_Return(tmp));
            Py_DECREF(tmp);
        }
    }
    PyObject *t = PyTuple_New(3);
    PyTuple_SetItem(t, 0, PyArray_Return(Gtau));
    PyTuple_SetItem(t, 1, PyArray_Return(Gerr));
    PyTuple_SetItem(t, 2, obs);
    return t;
}
