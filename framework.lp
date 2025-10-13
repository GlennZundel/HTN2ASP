
1 { occurs(A, T) : action(A) } 1 :- time(T), T < n.

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


#show occurs/2.
