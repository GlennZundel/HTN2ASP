

out_state(in(O, L), T+1) :- taskTBA(pickup(O, L), T).
out_state(armempty, T+1) :- taskTBA(pickup(O, L), T).
in_state(holding(O), T+1) :- taskTBA(pickup(O, L), T).


out_state(holding(O), T+1) :- taskTBA(putdown(O, L), T).
in_state(armempty, T+1) :- taskTBA(putdown(O, L), T).
in_state(in(O, L), T+1) :- taskTBA(putdown(O, L), T).


out_state(rloc(L1), T+1) :- taskTBA(move(L1, L2, D), T).
in_state(rloc(L2), T+1) :- taskTBA(move(L1, L2, D), T).


out_state(closed(D), T+1) :- taskTBA(open(L1, L2, D), T).


method_ag_pickup(T) :- 
    taskTBA(achieve_goals, T),
    not method_ag_move(T),
    not method_ag_open(T),
    not method_ag_finished(T).

method_ag_move(T) :- 
    taskTBA(achieve_goals, T),
    not method_ag_pickup(T),
    not method_ag_open(T),
    not method_ag_finished(T).

method_ag_open(T) :- 
    taskTBA(achieve_goals, T),
    not method_ag_pickup(T),
    not method_ag_move(T),
    not method_ag_finished(T).

method_ag_finished(T) :- 
    taskTBA(achieve_goals, T),
    not method_ag_pickup(T),
    not method_ag_move(T),
    not method_ag_open(T).

checked_state(ag_pickup_cond1(L, O), T) :-
    method_ag_pickup(T),
    room(L), package(O),
    in_state(rloc(L), T),
    in_state(in(O, L), T),
    not in_state(goal_in(O, L), T).


taskTBA(pickup_abstract(O), T) :-
    checked_state(ag_pickup_cond1(L, O), T).

taskTBA(release, T2) :-
    checked_state(ag_pickup_cond1(L, O), T),
    causable(pickup_abstract(O), T, T2),
    T2 >= T.

causable(achieve_goals, T, T3) :-
    checked_state(ag_pickup_cond1(L, O), T),
    causable(pickup_abstract(O), T, T2),
    causable(release, T2, T3),
    T3 >= T2, T2 >= T.

checked_state(ag_move_cond, T) :- method_ag_move(T).


taskTBA(move_abstract, T) :-
    checked_state(ag_move_cond, T).

taskTBA(achieve_goals, T2) :-
    checked_state(ag_move_cond, T),
    causable(move_abstract, T, T2),
    T2 >= T.

causable(achieve_goals, T, T3) :-
    checked_state(ag_move_cond, T),
    causable(move_abstract, T, T2),
    causable(achieve_goals, T2, T3),
    T3 >= T2, T2 >= T.

checked_state(ag_open_cond, T) :- method_ag_open(T).

taskTBA(open_abstract, T) :-
    checked_state(ag_open_cond, T).

taskTBA(achieve_goals, T2) :-
    checked_state(ag_open_cond, T),
    causable(open_abstract, T, T2),
    T2 >= T.


causable(achieve_goals, T, T3) :-
    checked_state(ag_open_cond, T),
    causable(open_abstract, T, T2),
    causable(achieve_goals, T2, T3),
    T3 >= T2, T2 >= T.

checked_state(ag_finished_cond, T) :- method_ag_finished(T).

causable(achieve_goals, T, T) :-
    checked_state(ag_finished_cond, T).


method_rel_putdown(T) :- 
    taskTBA(release, T),
    not method_rel_move(T),
    not method_rel_open(T).

method_rel_move(T) :- 
    taskTBA(release, T),
    not method_rel_putdown(T),
    not method_rel_open(T).

method_rel_open(T) :- 
    taskTBA(release, T),
    not method_rel_putdown(T),
    not method_rel_move(T).


checked_state(rel_putdown_cond1(L, O), T) :-
    method_rel_putdown(T),
    room(L), package(O),
    in_state(rloc(L), T),
    in_state(holding(O), T),
    in_state(goal_in(O, L), T).


taskTBA(putdown_abstract, T) :-
    checked_state(rel_putdown_cond1(L, O), T).

taskTBA(achieve_goals, T2) :-
    checked_state(rel_putdown_cond1(L, O), T),
    causable(putdown_abstract, T, T2),
    T2 >= T.

causable(release, T, T3) :-
    checked_state(rel_putdown_cond1(L, O), T),
    causable(putdown_abstract, T, T2),
    causable(achieve_goals, T2, T3),
    T3 >= T2, T2 >= T.

checked_state(rel_move_cond, T) :- method_rel_move(T).

taskTBA(move_abstract, T) :-
    checked_state(rel_move_cond, T).

taskTBA(release, T2) :-
    checked_state(rel_move_cond, T),
    causable(move_abstract, T, T2),
    T2 >= T.

causable(release, T, T3) :-
    checked_state(rel_move_cond, T),
    causable(move_abstract, T, T2),
    causable(release, T2, T3),
    T3 >= T2, T2 >= T.

checked_state(rel_open_cond, T) :- method_rel_open(T).


taskTBA(open_abstract, T) :-
    checked_state(rel_open_cond, T).

taskTBA(release, T2) :-
    checked_state(rel_open_cond, T),
    causable(open_abstract, T, T2),
    T2 >= T.

causable(release, T, T3) :-
    checked_state(rel_open_cond, T),
    causable(open_abstract, T, T2),
    causable(release, T2, T3),
    T3 >= T2, T2 >= T.

method_pickup(O, T) :- taskTBA(pickup_abstract(O), T).

checked_state(pickup_cond(O, L), T) :-
    method_pickup(O, T),
    room(L),
    in_state(rloc(L), T),
    in_state(in(O, L), T),
    in_state(armempty, T).

taskTBA(pickup(O, L), T) :-
    checked_state(pickup_cond(O, L), T).

method_putdown(T) :- taskTBA(putdown_abstract, T).

checked_state(putdown_cond(O, L), T) :-
    method_putdown(T),
    package(O), room(L),
    in_state(rloc(L), T),
    in_state(holding(O), T),
    in_state(goal_in(O, L), T).

taskTBA(putdown(O, L), T) :-
    checked_state(putdown_cond(O, L), T).

method_move(T) :- taskTBA(da , T).

checked_state(move_cond(L1, L2, D), T) :-
    method_move(T),
    room(L1), room(L2), roomdoor(D),
    in_state(rloc(L1), T),
    in_state(door(L1, L2, D), T),
    not in_state(closed(D), T).

taskTBA(move(L1, L2, D), T) :-
    checked_state(move_cond(L1, L2, D), T).

method_open(T) :- taskTBA(open_abstract, T).

checked_state(open_cond(L1, L2, D), T) :-
    method_open(T),
    room(L1), room(L2), roomdoor(D),
    in_state(rloc(L1), T),
    in_state(door(L1, L2, D), T),
    in_state(closed(D), T).

taskTBA(open(L1, L2, D), T) :-
    checked_state(open_cond(L1, L2, D), T).

causable(pickup_abstract(O), T, T+1) :-
    checked_state(pickup_cond(O, L), T),
    taskTBA(pickup(O, L), T).

causable(putdown_abstract, T, T+1) :-
    checked_state(putdown_cond(O, L), T),
    taskTBA(putdown(O, L), T).

causable(move_abstract, T, T+1) :-
    checked_state(move_cond(L1, L2, D), T),
    taskTBA(move(L1, L2, D), T).

causable(open_abstract, T, T+1) :-
    checked_state(open_cond(L1, L2, D), T),
    taskTBA(open(L1, L2, D), T).



