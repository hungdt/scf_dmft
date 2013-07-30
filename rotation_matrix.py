from numpy import *;

def generate_transform_mat(mat_file, L=[2]):
    tmp = genfromtxt(mat_file);
    ncell = len(tmp) / 3;
    ret = {};
    for l in L:
        ret[l] = [];
        for c in range(ncell):
            R = tmp[3*c:3*(c+1),:];
            ret[l].append(mat(rot_Ylm(l, R)));
    return ret;

def rot_mat(p):
    x1 = p[1] - p[0];
    y1 = p[2] - p[0];
    z1 = p[3] - p[0];
    x1 /= linalg.norm(x1); y1 /= linalg.norm(y1); z1 /= linalg.norm(z1);

    Rorig = array([x1,y1,z1])

    z1 = cross(x1,y1); x1 = cross(y1,z1);
    R = array([x1, y1, z1]);
    return Rorig, R;

def get_Euler_angles(R):
    # get Euler angle: b, a, g
    b = arccos(R[2,2]);
    if sin(b) > 1e-10:
        cosa = R[2,0]/sin(b);
        cosg = -R[0,2]/sin(b);
        if abs(abs(cosa) - 1) < 1e-4: cosa = cosa/abs(cosa);
        if abs(abs(cosg) - 1) < 1e-4: cosg = cosg/abs(cosg);
        a = arccos(cosa)  if R[2,1] >= 0 else 2*pi-arccos(cosa);
        g = arccos(cosg) if R[1,2] >= 0 else 2*pi-arccos(cosg);
    else:
        if R[0,1] >= 0: a = g = arccos(R[0,0])/2;
        if R[0,1] < 0 : a = g = (pi - arccos(R[0,0]))/2;
    return (a, b, g);

def rot_mat_Euler(a, b, g):
    mat = array([
        [cos(a)*cos(b)*cos(g)-sin(a)*sin(g), cos(a)*sin(g)+sin(a)*cos(b)*cos(g), -sin(b)*cos(g)],
        [-cos(a)*cos(b)*sin(g)-sin(a)*cos(g), cos(a)*cos(g)-sin(a)*cos(b)*sin(g), sin(b)*sin(g)],
        [cos(a)*sin(b), sin(a)*sin(b), cos(b)]]);
    return mat;

def dY(l, m1, m, b):
    binom = lambda a,b: math.factorial(a) / float(math.factorial(a-b)*math.factorial(b));
    ret = 0.;
    for k in range(max(0, m-m1), min(l-m1, l+m)+1): ret += (-1)**k * binom(l+m, k) * binom(l-m, l-m1-k) * cos(b/2)**(2*l-m1+m-2*k) * sin(b/2)**(2*k-m+m1);
    ret *= sqrt(math.factorial(l+m1) * math.factorial(l-m1) / float(math.factorial(l+m) * math.factorial(l-m))) * (-1)**(m1-m);
    return ret;

def rot_Ylm(l, R):
    (a, b, g) = get_Euler_angles(R);
#    print "Euler angles: a = %.4f, b = %.4f, g = %.4f"%(a,b,g);

    U = zeros((2*l+1, 2*l+1));

    # ylm1 = yl0 + yl1+ + yl1- + ...
    # Yl0
    u = zeros(2*l+1);
    u[0] = dY(l,0,0,b);
    for m1 in range(1, l+1): 
        u[2*m1-1] = 1/sqrt(2)*((-1)**m1*dY(l, m1, 0, b) + dY(l, -m1, 0, b))*cos(m1*a);
        u[2*m1]   = 1/sqrt(2)*((-1)**m1*dY(l, m1, 0, b) + dY(l, -m1, 0, b))*sin(m1*a);
    U[0,:] = u;

    # Ylm for m > 0
    for m in range(1, l+1):
        u = zeros(2*l+1);
        u[0] = (-1)**m * dY(l, 0, m, b) * cos(m*g) * sqrt(2);
        for m1 in range(1, l+1):
            u[2*m1-1] = (-1)**(m+m1)*dY(l,m1,m,b)*cos(m*g+m1*a) + (-1)**m*dY(l,-m1,m,b)*cos(m*g-m1*a);
            u[2*m1]   = (-1)**(m+m1)*dY(l,m1,m,b)*sin(m*g+m1*a) - (-1)**m*dY(l,-m1,m,b)*sin(m*g-m1*a);
        U[2*m-1,:] = u;

        u[0] = (-1)**(m+1) * dY(l, 0, m, b) * sin(m*g) * sqrt(2);
        for m1 in range(1, l+1):
            u[2*m1-1] = (-1)**(m+m1+1)*dY(l,m1,m,b)*sin(m*g+m1*a) + (-1)**(m+1)*dY(l,-m1,m,b)*sin(m*g-m1*a);
            u[2*m1]   = (-1)**(m+m1)*dY(l,m1,m,b)*cos(m*g+m1*a) - (-1)**m*dY(l,-m1,m,b)*cos(m*g-m1*a);
        U[2*m,:] = u;
    return mat(U);

