


%=======================================================================
%== ASSIGNMENT : hw4, Optimization (MAE 5930)
%== AUTHOR     : Jared Hansen
%== DUE        : Thursday, 10/17/2019
%=======================================================================
clear all; close all; clc;









%=======================================================================
%=======================================================================
%== PROBLEM 1
%=======================================================================
%=======================================================================
% Coefficients for objective function where x = [j,d,t,c,b,i,e] and
% the value of each element is either 0 or 1. For example, if j=1
% this means that we choose to pay Jenna her $30 for the 1-5PM slot.
% If d=0 this would mean that we elect not to take Doug up on the 
% 1-3PM slot for $18 (and so on for the rest of the guards).
c = [30, 18, 21, 38, 20, 22, 9];
% Specify which variables must be integers (all of them. Must be
% either 1 or 0 -- "working" or "not working", controlled by lb & ub)
intcon = (1:7);
% The matrix for inequality constraints. This is a binary encoding
% of each guard's availability. The first column in the matrix is
% Jenna's hours, the second col is Doug's hours, etc.
A(1,:) = [1, 1, 0, 0, 0, 0, 0];  % who is available during 1-2PM
A(2,:) = [1, 1, 0, 0, 0, 0, 0];  % who is available during 2-3PM
A(3,:) = [1, 0, 0, 0, 0, 0, 0];  % who is available during 3-4PM
A(4,:) = [1, 0, 1, 1, 0, 0, 0];  % who is available during 4-5PM
A(5,:) = [0, 0, 1, 1, 0, 1, 0];  % who is available during 5-6PM
A(6,:) = [0, 0, 1, 1, 1, 1, 0];  % who is available during 6-7PM
A(7,:) = [0, 0, 0, 1, 1, 1, 0];  % who is available during 7-8PM
A(8,:) = [0, 0, 0, 1, 1, 0, 1];  % who is available during 8-9PM
% The column vector for inequality constraints. This is representing
% the constraint that at least one guard must be on duty at all 
% times (it's negative so that we have <= instead of >= condition).
b = -ones(8,1);
% By restricting the lower and upper bound of our decision vector x
% to 0 and 1 respectively (along with specifying the "intcon" of all
% variables needing to be integers) we have ensured that our output
% is a vector of only 1's and 0's to show who is and isn't working.
lb = zeros(7,1);
ub = ones(7,1);

% Solve the problem using intlinprog
intLP_soln = intlinprog(c, intcon, -A, b, [], [], lb, ub);
disp("Problem 1 solution (Security Guards):")
intLP_soln











clear all; close all; clc;
%=======================================================================
%=======================================================================
%== PROBLEM 2
%=======================================================================
%=======================================================================
% Create vectors for mass, volume, and value coefficients from the table
m_coef =  [500, 1500, 2100, 600, 400];
v_coef =  [25, 15, 13, 20, 16];
value_coef = [50000; 60000; 90000; 40000; 30000];
% Coefficients for objective function where 
% x = [a1,b1,...,e1,a2,b2,...,e2,a3,b3,...,e3] where a1 is the number of
% A crates in the front segment, b3 is the number of B crates in the 
% back segment, c2 is the number of C crates in the middle segment, etc
c = [value_coef; value_coef; value_coef];
% Specify which variables must be integers (all of them).
intcon = (1:15);
% The matrix encoding inequality constraints.
A(1,:) = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0];  % no more than 12 A crates
A(2,:) = [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0];  % no more than 8  B crates
A(3,:) = [0,0,1,0,0,0,0,1,0,0,0,0,1,0,0];  % no more than 22 C crates
A(4,:) = [0,0,0,1,0,0,0,0,1,0,0,0,0,1,0];  % no more than 15 D crates
A(5,:) = [0,0,0,0,1,0,0,0,0,1,0,0,0,0,1];  % no more than 11 E crates
A(6,:) = [m_coef, zeros(1,10)];            % <= 8000  kg in front
A(7,:) = [zeros(1,5), m_coef, zeros(1,5)]; % <= 20000 kg in middle
A(8,:) = [zeros(1,10), m_coef];            % <= 6000  kg in back
A(9,:) = [v_coef, zeros(1,10)];            % <= 200  m^3 in front
A(10,:)= [zeros(1,5), v_coef, zeros(1,5)]; % <= 500  m^3 in middle
A(11,:)= [zeros(1,10), v_coef];            % <= 300  m^3 in back
A(12,:)= [m_coef, -m_coef, m_coef];        % massM >= (massF + massB)
A(13,:)= [-2*m_coef, m_coef, -2*m_coef];   % massM <= 2(massF + massB)
% The corresponding column vector for inequality constraints
b = [12; 8; 22; 15; 11; 8000; 20000; 6000; 200; 500; 300; 0; 0];
% Specify lower bounds for all values in x (upper bounds are 
% accounted for in the first 5 constraints in the matrix above).
lb = zeros(15,1);
% Solve the problem using intlinprog (-c since we're minimizing)
intLP_plane = intlinprog(-c, intcon, A, b, [], [], lb);
disp("Problem 2 solution (plane packing):")
intLP_plane
% Max value we can load onto a plane in this problem is $2,130,000
disp("Max value of a plane in this problem:")
c.' * intLP_plane













% HW - 4 ----- PROBLEM - 2

 

f4 = -1*[50 50 50 60 60 60 90 90 90 40 40 40 30 30 30]; %Value in $(000s)

b4 = [200 500 300 800 20000 6000 0 0];

A4 = [25 0 0 15 0 0 13 0 0 20 0 0 16 0 0

      0 25 0 0 15 0 0 13 0 0 20 0 0 16 0

      0 0 25 0 0 15 0 0 13 0 0 20 0 0 16

      500 0 0 1500 0 0 2100 0 0 600 0 0 400 0 0

      0 500 0 0 1500 0 0 2100 0 0 600 0 0 400 0

      0 0 500 0 0 1500 0 0 2100 0 0 600 0 0 400

      -1000 500 -1000 -3000 1500 -3000 -4200 2100 -4200 -1200 600 -1200 -800 400 -800

      500 -500 500 1500 -1500 1500 2100 -2100 2100 600 -600 600 400 -400 400

    ];

 

LB4 = [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]; % Lower bounds of Decision Var

UB4 = [12 12 12 8 8 8 22 22 22 15 15 15 11 11 11]; % Upper bounds of Decision Var

 

% Solution of the integer linear program (make all fifteen variables integer)

sol4 = intlinprog(f4,[1:15],A4,b4,[],[],LB4,UB4) ;

%Max Value

disp(-1*f4*sol4)















%==== JAKE'S CODE BELOW HERE ==============================================
%=========================================================================%
%=========================================================================%
clear all; clc;
%=========================================================================%
%-------------------------Homework 4 Problem 2----------------------------%
%=========================================================================%
c = 10000.*[5 5 5 6 6 6 9 9 9 4 4 4 3 3 3]; %Objective Function
A = zeros(11,15); %Consttraints matrix
A(1,:) = [1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]; %Crates of type A
A(2,:) = [0 0 0 1 1 1 0 0 0 0 0 0 0 0 0]; %Crates of type B
A(3,:) = [0 0 0 0 0 0 1 1 1 0 0 0 0 0 0]; %Crates of type C
A(4,:) = [0 0 0 0 0 0 0 0 0 1 1 1 0 0 0]; %Crates of type D
A(5,:) = [0 0 0 0 0 0 0 0 0 0 0 0 1 1 1]; %Crates of type E
A(6,:) = [5 0 0 15 0 0 21 0 0 6 0 0 4 0 0]*100; %Mass of crates in the front
A(7,:) = [25 0 0 15 0 0 13 0 0 20 0 0 6 0 0]; %Volume of crates in the front
A(8,:) = [0 5 0 0 15 0 0 21 0 0 6 0 0 4 0]*100; %Mass of crates in the middle
A(9,:) = [0 25 0 0 15 0 0 13 0 0 20 0 0 6 0]; %Volume of crates in the middle
A(10,:) = [0 0 5 0 0 15 0 0 21 0 0 6 0 0 4]*100; %Mass of crates in the back
A(11,:) = [0 0 25 0 0 15 0 0 13 0 0 20 0 0 6]; %Volume of crates in the back
A(12,:) = [5 -5 5 15 -15 15 21 -21 21 6 -6 6 4 -4 4]*100; %M_l >= M_f + M_k
A(13,:) = [-10 5 -10 -30 15 -30 -42 21 -42 -12 6 -12 -8 4 -8]*100; %M_l <= 2(M_f + M_k)
b = [12 8 22 15 11 8000 200 20000 500 6000 300 0 0]; %RHS inequality constraints
lb = zeros(15,1); %Lower bound
sol = intlinprog(-c,1:15,A,b,[],[],lb,[]) %Solution to integer program
%=========================================================================%


% Max value for Jake's solution: $2,130,000
maxVal_jake = (c * sol);
maxVal_jake
% It looks like Jake's solution violates the constraint that the volume
% of crates in the front be <= 200 m^3











