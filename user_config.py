import os;

src_dir = os.path.dirname(os.path.abspath(__file__));

# this is for the interface with the private CT-HYB solver 
# from Philipp Werner and Emanuel Gull
LatticeLibrary = '"/home/hungdt/works1/share/lattices.xml"';
ModelLibrary = '"/home/hungdt/works1/share/models.xml"';
Model = '"hung_multiorbital"';
mpirun = "$MPIEXEC -x PYTHONPATH";
parm2xml = "parameter2xml";
solver_matrix = "/home/hungdt/works/code/DMFT/Matrix/MPI_dca";

# for segment solvers in ALPS code
# the path for alpspython, usually in $ALPS_ROOT_DIR/bin
alpspython = "/opt/alps_svn/bin/alpspython"
solver_segment = "sh %s %s/alps_interface.py"%(alpspython, src_dir);

# for TRIQS code
# the path for pytriqs, usually in $TRIQS_ROOT_DIR/bin
pytriqs = "pytriqs"
solver_triqs = "%s %s/triqs_interface.py"%(pytriqs, src_dir);

# this is for the interface with the private segment solver from Michel Ferrero
solver_segment_triqs = "pytriqs-0.8 %s/triqs_segment_interface.py"%src_dir;
