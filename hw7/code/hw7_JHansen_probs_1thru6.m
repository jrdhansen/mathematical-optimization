%=======================================================================
%== ASSIGNMENT : hw7, Optimization (MAE 5930)
%== AUTHOR     : Jared Hansen
%== DUE        : Tuesday, 11/26/2019
%=======================================================================
clear all; close all; clc;








%{
%=======================================================================
%== PROBLEM 1
%======================================================================= 

% Initial guess (we know x* = [0,0], so we pick a close point.)
x0 = [0.5; 0.3];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 1 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = (x1)^2 + (x2)^2;
end
% Problem 1 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = x1 + x2 - 2;
cin = (x1)^2 - x2 - 4;
ceq = [];
end
%}







%{
clear all; close all; clc;

%=======================================================================
%== PROBLEM 2
%======================================================================= 

% Initial guess (we know x* = [4,0], so we pick a close point.)
x0 = [4.2; 0.3];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 2 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = (x1)^2 + (x2)^2;
end
% Problem 2 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = x1 -10;
cin = -x1 + (x2)^2 +4;
ceq = [];
end
%}











%{
clear all; close all; clc;

%=======================================================================
%== PROBLEM 3
%======================================================================= 

% Initial guess (we know x* = [3,1], so we pick a close point.)
% NOTE: not even giving the correct answer could get fmincon to give the
%       right answer.
x0 = [3.0; 1.0];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 3 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = (x1)^2 + (x2)^2;
end
% Problem 3 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = 4 -x1 - (x2)^2 ;
cin = 3*x2 -x1 ;
cin = -3*x2  -x1 ;
ceq = [];
end
%}












%{
clear all; close all; clc;

%=======================================================================
%== PROBLEM 4
%======================================================================= 

% Initial guess...answer DNE, so doesn't really matter what we put in.
x0 = [-10; 14];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 4 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = x1 * x2 ;
end
% Problem 4 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = -x1 - x2 + 2 ;
cin = x1 - x2 ;
ceq = [];
end
%}










clear all; close all; clc;

%=======================================================================
%== PROBLEM 5
%======================================================================= 

% Initial guess. We know x* = [1,0] so we'll pick x0 close to that.
% NOTE: even giving an initial guess that is the answer can't get
%       fmincon to converge to the correct answer.
%       However, I accidentally made g3(x) equality instead of ineq
%       and it did give the correct answer.
x0 = [1.0; 0.0];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 5 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = -x1  ;
end
% Problem 5 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = -x1 ;
cin = -x2 ;
cin = x2 + (x1 - 1)^3 ;
ceq = [] ;
end











%{
clear all; close all; clc;

%=======================================================================
%== PROBLEM 6
%======================================================================= 

% Initial guess. We know x* = [0,0] so we'll pick x0 close to that.
x0 = [0.005; 0.005];
 
% Solve the problem using fmincon.
[x,f] = fmincon(@obj,x0,[],[],[],[],[],[],@con)
 
% Problem 6 Objective function
function J = obj(x)
x1 = x(1);
x2 = x(2);
J  = 2*(x1^2) - (x2^2) ;
end
% Problem 6 Constraint functions
function [cin,ceq] = con(x)
x1 = x(1);
x2 = x(2);
cin = (x1^2)*x2 - (x2^3) ;
ceq = [];
end
%}

