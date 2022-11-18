%=======================================================================
%== ASSIGNMENT : hw5, Optimization (MAE 5930)
%== AUTHOR     : Jared Hansen
%== DUE        : Thursday, 10/24/2019
%=======================================================================


clear all; close all; clc
%=======================================================================
%== PROBLEM 1
%=======================================================================





%=======================================================================
%== 1A: plot the functions and visually confirm that the optimal
%==     point is approximately 3
%=======================================================================
f = @(x) x^2;
g = @(x) x^2*sin(x);
fplot(f,[1,12], 'linewidth', 2), hold on, grid on;
fplot(g,[1,12], '--or');
title('Functions f and g on the interval x in [1,12]');
xlabel('x');
ylabel('y (f(x) and g(x))');




%=======================================================================
%== 1B: solve the problem using fmincon, x0 in {1,5,9}
%=======================================================================

% NOTE: the nonlinear constraint must be defined in its own .m file.
%       I've only included it here for visualizing's sake. This code
%       wouldn't run as-is. Put the 4 lines below in their own file.
%function [c,ceq] = ineqConFun(x)
%c = x^2*sin(x);   % defining nonlinear ineq. constraint function
%ceq = [];         % defining nonlinear, eq. constraint function (n/a)
%end;
obj = @(x) x^2 ;  % defining the objective function
nonlcon = @ineqConFun;  % specifying nonlinear constraints
% Solutions for different guesses of x0: 1, 5, and 9
xOptim_guess1 = fmincon(obj,1,[],[],[],[],1,12,nonlcon);
xOptim_guess5 = fmincon(obj,5,[],[],[],[],1,12,nonlcon);
xOptim_guess9 = fmincon(obj,9,[],[],[],[],1,12,nonlcon);





%=======================================================================
%== 1D: solve the MILP using 20, 50, and 100 nodes.
%=======================================================================

% CREATE THE NODE POINTS
n = 100;                % How many nodes ?
x = linspace(1,12,n);  % Creating discrete x    values
y = x.^2;              % Creating discrete f(x) values
z = x.^2.*sin(x);      % Creating discrete g(x) values

% SET UP THE OPTIMIZATION PROBLEM
prob = optimproblem('ObjectiveSense','minimize');
 
% DEFINE VARIALBLES (BINARIES AND WEIGHTS)
b = optimvar('b',n-1,1, 'Type','integer',   'LowerBound',0,'UpperBound',1);
w = optimvar('w',n,1,   'Type','continuous','LowerBound',0,'UpperBound',1);
 
% DEFINE THE OBJECTIVE FUNCTION
prob.Objective = sum( y*w );
 
% DEFINE CONSTRAINTS
prob.Constraints.bsum = sum(b) == 1;     % all binaries must sum to 1
prob.Constraints.wsum = sum(w) == 1;     % all weights  must sum to 1
prob.Constraints.gIneq = sum(z*w) <= 0;  % discretization of g(x) <= 0 
 
% CREATE SOS (Special Ordered Set) CONSTRAINTS
SOS = zeros(n,n-1);
SOS([1:n+1:end]) = 1;
SOS([2:n+1:end]) = 1;
prob.Constraints.sos = SOS*b >= w;
 
% DETERMINE THE SOLUTION
sol = solve(prob)
xsol = x*sol.w
ysol = y*sol.w
 
% PLOT RESULTS
plot(x,y,'+'), grid on, hold on
plot(x,z,'*'), grid on
plot(xsol,ysol,'ko','markerfacecolor','k')
xlabel('x'), ylabel('f(x)=o and g(x)=*')
title(strcat(['Piecewise Linear Approx solution with ', int2str(n), ' nodes']));

% EXAMINE SOLUTION
disp(strcat(['Optimal x, f(x) for ', int2str(n), ' nodes']));
xsol, ysol


















clear all; close all; clc
%=======================================================================
%== PROBLEM 2
%=======================================================================


%== DRAWING A MAP OF THE UNITED STATES WITH 
figure;
load('usborder.mat','x','y','xx','yy');
rng(3,'twister') % makes a plot with stops in Maine & Florida, and is reproducible
nStops = 200; % you can use any number, but the problem size scales as N^2
stopsLon = zeros(nStops,1); % allocate x-coordinates of nStops
stopsLat = stopsLon; % allocate y-coordinates
n = 1;
while (n <= nStops)
    xp = rand*1.5;
    yp = rand;
    if inpolygon(xp,yp,x,y) % test if inside the border
        stopsLon(n) = xp;
        stopsLat(n) = yp;
        n = n+1;
    end
end
plot(x,y,'Color','red'); % draw the outside border
hold on
% Add the stops to the map
plot(stopsLon,stopsLat,'*b')












