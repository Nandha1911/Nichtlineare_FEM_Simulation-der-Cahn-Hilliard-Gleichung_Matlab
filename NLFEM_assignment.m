%% Constant values
dt = 1.2;
tend = input("Enter the time step for convergence :\n");  % Assigned as t > 6000
t = (0:dt:tend);
cl = 0.25;
cu = 0.75;
Mo = 0.3;
l = 10;  % Function length
h = input("enter the size of the element :\n");  % Assigned as h = 0.1
elements = 0:h:l;
elements1 = transpose(elements);
nodes = ((length(elements)-1)+1)*2;

% Initialize concentration field
cinitial = cvalues(cl, cu, nodes);
cglobold = cinitial;

% Material and model parameters
Psinot = 0.055;
chi = 2.7;
lambda = 0.01375;
x = [0, h];

z = 1;  % Columns for assembly
zz = 4;

% Gauss quadrature values (4-point integration)
xi = [-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053];
w = [0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454];

%% Hermite Shape function (anonymous)
% Interpolation functions for Hermite elements (cubic)
Nherm = @(xi) [1/4*((2+xi)*((xi-1)^2)), h/8*((xi+1)*((xi-1)^2)), ...
               1/4*((2-xi)*((xi+1)^2)), h/8*((xi-1)*((xi+1)^2))];

%% Lagrangian shape function
% Linear Lagrange shape functions for Jacobian computation
Nlagr = [1/2*(1 - xi), 1/2*(1 + xi)];
DNlagr = [-1/2, 1/2];  % First derivative

Jac = DNlagr * transpose(x);  % Jacobian matrix

%% Newton-Raphson counter initialization
Newrap_counter = zeros(length(elements), 1);

% Main time loop
Rglobold = zeros(nodes, 1);
cglobnew = cglobold;

%% Assembly matrix logic
for m = 1:length(t)
    ii = 0;
    while true
        z = 1; zz = 4; aa = 4;
        Kglob = zeros(nodes, nodes);
        Rglobnew = zeros(nodes, 1);

        % Loop over all elements to assemble global matrices
        for n = 1:(length(elements) - 1)
            A = zeros(aa, ((length(elements) - 1) + 1)*(aa / 2));
            A2 = A(1:aa, z:zz);
            cc = A2 + diag(ones(1, aa));
            A(1:aa, z:zz) = cc;

            ccapold = A * cglobold;
            ccapnew = A * cglobnew;

            % Element stiffness matrix and residual vector
            K = transpose(A) * ktm(Nherm, ccapnew, xi, Jac, h, Psinot, chi, lambda, Mo, w, dt) * A;
            Rcapele = transpose(A) * Res(Nherm, ccapnew, ccapold, xi, Jac, h, Psinot, chi, lambda, Mo, w, dt);

            Kglob = Kglob + K;
            Rglobnew = Rglobnew + Rcapele;

            z = z + 2;
            zz = zz + 2;
        end

        % Newton-Raphson update
        dccap = Kglob \ (-Rglobnew);
        cglobnew = cglobnew + dccap;
        ii = ii + 1;  % Iteration count

        disp(norm(dccap, inf));
        disp(1e-5 * norm(cglobnew, inf));
        disp(norm(Rglobnew, inf));
        disp(0.005 * max(norm(Rglobold, inf), 1e-8));

        % Convergence check
        if (norm(dccap, inf) < 1e-5 * (norm(cglobnew, inf)) && ...
            norm(Rglobnew, inf) < 0.005 * max(norm(Rglobold, inf), 1e-8))
            break;
        end
        Rglobold = Rglobnew;
    end
    cglobold = cglobnew;

    % Energy calculation
    p = 1; pp = 4; bb = 4;
    for oo = 1:(length(elements) - 1)
        A1 = zeros(bb, ((length(elements) - 1) + 1)*(bb / 2));
        A3 = A1(1:bb, p:pp);
        kk = A3 + diag(ones(1, bb));
        A1(1:bb, p:pp) = kk;
        ccapnew1 = A1 * cglobnew;

        Penergy = 0;
        for o = 1:4
            Psifin = Psiverif(xi, o, Nherm, ccapnew1, h, Jac, chi, lambda, Psinot, w);
            Penergy = Penergy + Psifin;
        end
        energy(oo) = Penergy;
        p = p + 2;
        pp = pp + 2;
    end
    Newrap_counter(m) = ii;
end

% Plotting
figure(1);
plot((0:h:l), cglobnew(1:2:end));
ylim([0 1]);
xlabel("Length (in mm)");
ylabel("Concentration");
hold on;
plot((0:h:l), cinitial(1:2:end));
ylim([0 1]);
legend("Initial concentration val", "Attained concentration val");

figure(2);
plot(energy);
title("Free energy curve");
xlabel("Time in sec");
ylabel("Free energy");

figure(3);
plot(Newrap_counter);
title("Newton Raphson Iterations");
xlabel("Time in sec");
ylabel("No. of Iterations");


%% --- Function Definitions --- %%

function R = Res(Nherm, ccapnew, ccapold, xi, Jac, h, Psinot, chi, lambda, Mo, w, dt)
% Computes the residual vector for a single element
R = zeros(4, 1);
for i = 1:4
    Nherm1 = Nherm(xi(i));
    c = Nherm1 * ccapnew;
    cdash = B(xi, i, Jac, h) * ccapnew;
    cdodash = G(xi, i, Jac, h) * ccapnew;

    Q1 = transpose(Nherm1) * Nherm1 * ((ccapnew - ccapold) / dt);
    Q2 = mobility(c, Mo) * freeenergy2(c, Psinot, chi) * cdash * transpose(B(xi, i, Jac, h));
    Q3 = mobility1(c, Mo) * lambda * cdodash * cdash * transpose(B(xi, i, Jac, h));
    Q4 = mobility(c, Mo) * lambda * cdodash * transpose(G(xi, i, Jac, h));
    Q = Q1 + Q2 + Q3 + Q4;

    temp2 = (w(i) * det(Jac)) * Q;
    R = temp2 + R;
end
end

function Kt = ktm(Nherm, ccapnew, xi, Jac, h, Psinot, chi, lambda, Mo, w, dt)
% Computes the element stiffness matrix using Hermite basis and model parameters
Kt = zeros(4, 4);
for i = 1:4
    Nherm1 = Nherm(xi(i));
    c = Nherm1 * ccapnew;
    cdash = B(xi, i, Jac, h) * ccapnew;
    cdodash = G(xi, i, Jac, h) * ccapnew;

    k1 = (transpose(Nherm1) * Nherm1) / dt;

    k2 = (mobility1(c, Mo) * freeenergy2(c, Psinot, chi) * cdash * (transpose(B(xi, i, Jac, h)) * Nherm1)) + ...
         (mobility(c, Mo) * freeenergy3(c, Psinot) * cdash * (transpose(B(xi, i, Jac, h)) * Nherm1)) + ...
         (mobility(c, Mo) * freeenergy2(c, Psinot, chi) * (transpose(B(xi, i, Jac, h)) * B(xi, i, Jac, h)));

    k3 = lambda * mobility2(c, Mo) * cdodash * cdash * (transpose(B(xi, i, Jac, h)) * Nherm1) + ...
         lambda * mobility1(c, Mo) * cdash * (transpose(B(xi, i, Jac, h)) * G(xi, i, Jac, h)) + ...
         lambda * mobility1(c, Mo) * cdodash * (transpose(B(xi, i, Jac, h)) * B(xi, i, Jac, h));

    k4 = lambda * mobility1(c, Mo) * cdodash * (transpose(G(xi, i, Jac, h)) * Nherm1) + ...
         lambda * mobility(c, Mo) * (transpose(G(xi, i, Jac, h)) * G(xi, i, Jac, h));

    temp3 = det(Jac) * w(i) * (k1 + k2 + k3 + k4);
    Kt = temp3 + Kt;
end
end

% Mobility and derivatives
function M = mobility(c, Mo)
% Concentration-dependent mobility
M = (3125 / 432) * Mo * (c^2) * ((1 - c)^3);
end

function dM = mobility1(c, Mo)
% First derivative of mobility
dM = (3125 / 432) * Mo * (2 * c - 9 * c^2 + 12 * c^3 - 5 * c^4);
end

function d2M = mobility2(c, Mo)
% Second derivative of mobility
d2M = (3125 / 432) * Mo * (2 - 18 * c + 36 * c^2 - 20 * c^3);
end

% Free energy derivatives
function d2Psi = freeenergy2(c, Psinot, chi)
% Second derivative of free energy function
d2Psi = Psinot * ((1 / c) + (1 / (1 - c)) - 2 * chi);
end

function d3Psi = freeenergy3(c, Psinot)
% Third derivative of free energy
d3Psi = Psinot * ((1 / ((c - 1)^2)) - (1 / (c^2)));
end

% Derivatives of Hermite shape functions
function B = B(xi, i, Jac, h)
% First derivative of Hermite shape functions
DNherm = [(3/4)*(xi(i)^2 - 1), (h/8)*(3*xi(i)^2 - 2*xi(i) + 1), ...
          (-3/4)*(xi(i)^2 - 1), (h/8)*(3*xi(i)^2 + 2*xi(i) - 1)];
B = DNherm * inv(Jac);
end

function G = G(xi, i, Jac, h)
% Second derivative of Hermite shape functions
DNhermsqr = [3/2 * xi(i), (h/4)*(3*xi(i) - 1), ...
            -3/2 * xi(i), (h/4)*(3*xi(i) + 1)];
G = DNhermsqr * inv(Jac) * inv(Jac);
end

% Initial concentration function
function cglobold = cvalues(cl, cu, nodes)
% Initializes the global concentration field
for i = 1:(nodes + 1)
    if mod(i, 2) ~= 0
        if i < (nodes / 2)
            cglobold(i, 1) = cl;
        else
            cglobold(i, 1) = cu;
        end
    end
end
cglobold(end) = [];
end

% Verification: compute total free energy
function Psi = Psiverif(xi, o, Nherm, ccapnew1, h, Jac, chi, lambda, Psinot, w)
% Computes free energy (chemical + gradient) for verification
Nherm1 = Nherm(xi(o));
c = Nherm1 * ccapnew1;
cdash = B(xi, o, Jac, h) * ccapnew1;
Psigiv = Psinot * (c * log(c) + (1 - c) * log(1 - c) + chi * c * (1 - c));
Psi = (w(o) * det(Jac)) * Psigiv + (1/2) * lambda * (abs(cdash)^2);
end
