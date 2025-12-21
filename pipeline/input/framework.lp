time(0..20).

in_state(A, T+1) :-
    time(T),
    in_state(A, T),
    not out_state(A, T+1).

0 { taskTBA(A, T) : action(A) } 1 :- time(T).

:- not plan_found.

#show taskTBA/2.