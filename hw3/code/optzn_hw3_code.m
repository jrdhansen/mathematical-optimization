


%=======================================================================
%== ASSIGNMENT : hw3, Optimization (MAE 5930)
%== AUTHOR     : Jared Hansen
%== DUE        : Thursday, 10/03/2019
%=======================================================================
clear all; close all; clc;
rng(1776)









%=======================================================================
%=======================================================================
%== PROBLEM 1
%=======================================================================
%=======================================================================

% Coefficients for objective function where x = [xb,xf,xe,xp,xd,xl,y] is
% our vector of decision variables. (We only care about the last
% element of the vector, y=completion time, so we set all other 
% coefficients to 0).
f = [0, 0, 0, 0, 0, 0, 1];
% The matrix for inequality constraints
A(1,:)  = [1, 0, 0, 0, 0, 0, -1];
A(2,:)  = [0, 1, 0, 0, 0, 0, -1];
A(3,:)  = [0, 0, 1, 0, 0, 0, -1];
A(4,:)  = [0, 0, 0, 1, 0, 0, -1];
A(5,:)  = [0, 0, 0, 0, 1, 0, -1];
A(6,:)  = [0, 0, 0, 0, 0, 1, -1];
A(7,:)  = [1,-1, 0, 0, 0, 0,  0];
A(8,:)  = [1, 0, 0, 0, 0,-1,  0];
A(9,:)  = [0, 1,-1, 0, 0, 0,  0];
A(10,:) = [0, 1, 0,-1, 0, 0,  0];
A(11,:) = [0 ,0, 1, 0,-1, 0,  0];
A(12,:) = [0, 0, 0, 1,-1, 0,  0];
% The column vector for inequality constraints
b = [-3; -2; -3; -4; -1; -2; -3; -3; -2; -2; -3; -4];
% The vector bounding x on the low side
lb = [0; 0; 0; 0; 0; 0; 0];
% The vector bounding x on the high side
ub = [inf; inf; inf; inf; inf; inf; inf];
% Solve the problem using linprog
linprog_soln = linprog(f, A, b, [], [], lb, ub);
disp("Problem 1 solution (construction):")
linprog_soln














clear all; close all; clc;
rng(1776)
%=======================================================================
%=======================================================================
%== PROBLEM 2
%=======================================================================
%=======================================================================

% Coefficients for the objective function (piping costs during each
% hour of the facility's operation)
f = [30, 40, 35, 45, 38, 50];
% The matrix for inequality constraints
A(1,:)  = [ 0,  0,  0,  0,  0, 0];
A(2,:)  = [-1,  0,  0,  0,  0, 0];
A(3,:)  = [-1, -1,  0,  0,  0, 0];
A(4,:)  = [-1, -1, -1,  0,  0, 0];
A(5,:)  = [-1, -1, -1, -1,  0, 0];
A(6,:)  = [-1, -1, -1, -1, -1, 0];
A(7,:)  = [ 1,  0,  0,  0,  0, 0];
A(8,:)  = [ 1,  1,  0,  0,  0, 0];
A(9,:)  = [ 1,  1,  1,  0,  0, 0];
A(10,:) = [ 1,  1,  1,  1,  0, 0];
A(11,:) = [ 1,  1,  1,  1,  1, 0];
A(12,:) = [ 1,  1,  1,  1,  1, 1];
% The column vector for inequality constraints
b = [700; 460; -140; -340; -640; -1540; 300; 540; 1140; 1340; 1640; 2540];
% The matrix for equality constraint
Aeq(1,:) = [1, 1, 1, 1, 1, 1];
% The column vector for the equality constraint
beq = [2540];
% The vector bounding x on the low side
lb = [0; 0; 0; 0; 0; 0];
% The vector bounding x on the high side
ub = [300; 1000; 1000; 1000; 1000; 1000];
% Solve the problem using linprog
linprog_soln_2 = linprog(f, A, b, Aeq, beq, lb, ub);
disp("Problem 2 solution (chemicals):")
linprog_soln_2



















clear all; close all; clc;
rng(1776)
%=======================================================================
%=======================================================================
%== PROBLEM 3
%=======================================================================
%=======================================================================

% Coefficients for the objective function 
f = [1, -1];
% The matrix for inequality constraints
A(1,:) = [1, 1];
A(2,:) = [-1, 2];
A(3,:) = [1, -3];
A(4,:) = [-1, 0];
A(5,:) = [0, -1];
A(6,:) = [1, 0];
A(7,:) = [0, 1];
% The column vector for inequality constraints
b = [1; 2; 3; 1; inf; inf; inf;];
% Solve the problem using linprog
linprog_soln_3 = linprog(f, A, b, [], [], [], []);
disp("Problem 3 solution (should be [-1,0.5]):")
linprog_soln_3


lamb = [0; 1/2; 0; 1/2; 0; 0; 0];
x = [-1; 0.5];






