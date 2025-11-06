
time(0..n).

0 { occurs(A, T) : action(A) } 1 :- time(T), T < n.

holds(F, T+1) :-
    holds(F, T),
    occurs(A, T),
    not deletes(A, F),
    time(T), T < n.

holds(F, T+1) :-
    occurs(A, T),
    adds(A, F),
    time(T), T < n.

:- occurs(A, T), precondition(A, F), not holds(F, T).
:- occurs(A, T), neg_precondition(A, F), holds(F, T).
:- occurs(A, T), not in_plan(A, T).


precondition_violated(M, T) :- 
    method_precondition(M, PosCond), 
    time(T),
    not holds(PosCond, T).

precondition_violated(M, T) :- 
    neg_method_precondition(M, NegCond), 
    time(T),
    holds(NegCond, T).

method_precondition_satisfied(M, T) :- 
    method(M), time(T), 
    not precondition_violated(M, T).

in_plan(A, T+1):- in_plan(A, T), time(T), T < n, not apply_method(A, T). 
in_plan(A, 0) :- initial_task(A).

apply_method(M,C,T) :- decomposes(M,C), method_precondition_satisfied(M,T), in_plan(C,T).
in_plan(A,T) :- apply_method(M,C,T), subtask(M,N,A).




%ordering subtasks
:- occurs(A1, T1), occurs(A2, T2), 
   subtask(M, N1, A1), subtask(M, N2, A2), 
   N1 < N2, T1 > T2.

%in case n is larger than minimum plan length -> optimize
#minimize { 1@1,T,A : occurs(A,T) }.
#show occurs/2.
#show apply_method/3.