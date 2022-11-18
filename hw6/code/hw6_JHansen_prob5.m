%=======================================================================
%== ASSIGNMENT : hw6, Optimization (MAE 5930)
%== AUTHOR     : Jared Hansen
%== DUE        : Wednesday, 11/06/2019
%=======================================================================
clear all; close all; clc;









%=======================================================================
%== PROBLEM 5
%=======================================================================
% Solve the problem using MATLAB's quadprog.

% Define the objective function (here we'll do |x|^2 since it's easier
% to define, and is the same as minimizing |x|)
% multiply by 2 due to Matlab's formulation with the 1/2 in front
H = 2 * eye(10);   
f = zeros(10,1)';
% Matrix Aeq and vector beq defining equality constraints
Aeq =[ 4 4 4  9 10 10 3 6 0 1;
       7 5 6  9 5  7  1 1 5 8;
       0 4 1  1 7  3  0 6 7 4;
       3 7 2  0 3  8  7 7 5 2;
       1 2 8  2 7  1  2 1 9 9;
       1 9 10 9 8  4  3 4 6 3;
       2 0 3  1 0  9  5 7 9 8;
       3 7 7  4 8  3  1 4 1 7];
beq = [9 6 8 3 3 9 4 10]';

soln = quadprog(H,f,[],[],Aeq,beq);
disp("Matlab solution using QUADPROG")
soln



