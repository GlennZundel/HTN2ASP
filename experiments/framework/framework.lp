time(0..25).

in_state(A, T+1) :-
    time(T),
    atom(A),
    in_state(A, T),
    not out_state(A, T+1).

:- not plan_found.

#show taskTBA/2.
