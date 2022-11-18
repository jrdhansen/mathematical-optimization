%Step 2: Write a file confuneq.m for the nonlinear constraint.
%=======================================================================
%function [c,ceq] = nlcon(x)
% Nonlinear inequality constraints
%c = x^2*sin(x);
% Nonlinear equality constraints (we have none)
%ceq = [];


function [c,ceq] = ineqConFun(x)
c = x^2*sin(x);
ceq = [];
end