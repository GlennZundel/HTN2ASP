#program step(t).
time(t).

in_state(A, t) :-
    in_state(A, t-1),
    not out_state(A, t).

#program check(t).
#external query(t).
:- query(t), not plan_found(t).

#show taskTBA/2.


