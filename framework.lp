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


has_pre_meth(M) :- method_precondition(M, _). 
has_neg_pre_meth(M) :- neg_method_precondition(M, _).
has_all_pre_meth(M) :- has_neg_pre_meth(M), has_pre_meth(M).

in_plan(A, T+1):- in_plan(A, T), time(T), T < n. 


in_plan(A, 0) :- initial_task(A).
in_plan(A, T) :- compound(C), decomposes(M, C), has_all_pre_meth(M), method_precondition(M, D), holds(D, T), neg_method_precondition(M, E), not holds(E, T), in_plan(C, T), subtask(M, _, A).
in_plan(A, T) :- compound(C), decomposes(M, C), has_pre_meth(M), method_precondition(M, D), holds(D, T), in_plan(C, T), subtask(M, _, A).
in_plan(A, T) :- compound(C), decomposes(M, C), has_neg_pre_meth(M), neg_method_precondition(M, D), not holds(D, T), in_plan(C, T), subtask(M, _, A).
in_plan(A, T) :- compound(C), decomposes(M, C), not has_pre_meth(M), not has_neg_pre_meth(M), in_plan(C, T), subtask(M, _, A).

:- occurs(A1, T1), occurs(A2, T2), 
   subtask(M, N1, A1), subtask(M, N2, A2), 
   N1 < N2, T1 > T2.



#minimize { T,A : occurs(A,T) }.
#show occurs/2.
