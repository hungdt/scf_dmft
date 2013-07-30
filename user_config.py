mpirun = "mpirun";

# for matrix solver
LatticeLibrary = '"/home/7fz/works/share/alps/lattices.xml"';
ModelLibrary = '"/home/7fz/works/share/alps/models.xml"';
Model = '"hung_multiorbital"';
parm2xml = "parameter2xml";
solver_matrix = "/home/7fz/works/code/DMFT/Matrix/MPI_dca.intel";

# for segment solvers
solver_segment = "sh /home/7fz/apps/alps_svn/gcc/bin/alpspython /home/7fz/works/code/DMFT/scf/alps_interface.py";

# for TRIQS
solver_triqs = "pytriqs /home/7fz/works/code/DMFT/scf/triqs_interface.py";
