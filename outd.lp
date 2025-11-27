% Primitive tasks (actions)

causable(pickup(OBJ, LOC), T, T+1) :- taskTBA(pickup(OBJ, LOC), T), package(OBJ), room(LOC).
out_state(in(OBJ, LOC), T+1) :- taskTBA(pickup(OBJ, LOC), T).
out_state(armempty, T+1) :- taskTBA(pickup(OBJ, LOC), T).
in_state(holding(OBJ), T+1) :- taskTBA(pickup(OBJ, LOC), T).

causable(putdown(OBJ, LOC), T, T+1) :- taskTBA(putdown(OBJ, LOC), T), package(OBJ), room(LOC).
out_state(holding(OBJ), T+1) :- taskTBA(putdown(OBJ, LOC), T).
in_state(armempty, T+1) :- taskTBA(putdown(OBJ, LOC), T).
in_state(in(OBJ, LOC), T+1) :- taskTBA(putdown(OBJ, LOC), T).

causable(move(LOC1, LOC2, D), T, T+1) :- taskTBA(move(LOC1, LOC2, D), T), room(LOC1), room(LOC2), roomdoor(D).
in_state(rloc(LOC2), T+1) :- taskTBA(move(LOC1, LOC2, D), T).
out_state(rloc(LOC1), T+1) :- taskTBA(move(LOC1, LOC2, D), T).

causable(open(LOC1, LOC2, D), T, T+1) :- taskTBA(open(LOC1, LOC2, D), T), room(LOC1), room(LOC2), roomdoor(D).
out_state(closed(D), T+1) :- taskTBA(open(LOC1, LOC2, D), T).

% Method selection rules

release_putdown_abstract(LOC, OBJ, T) :- taskTBA(release, T), package(OBJ), room(LOC), not release_move(T), not release_open(T).
release_move(T) :- taskTBA(release, T), not release_putdown_abstract(_, _, T), not release_open(T).
release_open(T) :- taskTBA(release, T), not release_putdown_abstract(_, _, T), not release_move(T).

achieve_goals_pickup(LOC, OBJ, T) :- taskTBA(achieve_goals, T), package(OBJ), room(LOC), not achieve_goals_move(T), not achieve_goals_open(T), not finished(T).
achieve_goals_move(T) :- taskTBA(achieve_goals, T), not achieve_goals_pickup(_, _, T), not achieve_goals_open(T), not finished(T).
achieve_goals_open(T) :- taskTBA(achieve_goals, T), not achieve_goals_pickup(_, _, T), not achieve_goals_move(T), not finished(T).
finished(T) :- taskTBA(achieve_goals, T), not achieve_goals_pickup(_, _, T), not achieve_goals_move(T), not achieve_goals_open(T).

newmethod22(OBJ, LOC, T) :- taskTBA(pickup_abstract(OBJ), T), package(OBJ), room(LOC).

newmethod23(OBJ, LOC, T) :- taskTBA(putdown_abstract, T), package(OBJ), room(LOC).

newmethod24(LOC1, LOC2, D, T) :- taskTBA(move_abstract, T), room(LOC1), room(LOC2), roomdoor(D).

newmethod25(LOC1, LOC2, D, T) :- taskTBA(open_abstract, T), room(LOC1), room(LOC2), roomdoor(D).

m_pickup_wrapper(OBJ, LOC, T) :- taskTBA(pickup_wrapper(OBJ, LOC), T), package(OBJ), room(LOC).

m_putdown_wrapper(OBJ, LOC, T) :- taskTBA(putdown_wrapper(OBJ, LOC), T), package(OBJ), room(LOC).

m_move_wrapper(LOC1, LOC2, D, T) :- taskTBA(move_wrapper(LOC1, LOC2, D), T), room(LOC1), room(LOC2), roomdoor(D).

m_open_wrapper(LOC1, LOC2, D, T) :- taskTBA(open_wrapper(LOC1, LOC2, D), T), room(LOC1), room(LOC2), roomdoor(D).

% Subtask rules

taskTBA(putdown_abstract, T) :- release_putdown_abstract(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ).
taskTBA(achieve_goals, T2) :- release_putdown_abstract(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ), causable(putdown_abstract, T, T2), T2 >= T.

taskTBA(move_abstract, T) :- release_move(T).
taskTBA(release, T2) :- release_move(T), causable(move_abstract, T, T2), T2 >= T.

taskTBA(open_abstract, T) :- release_open(T).
taskTBA(release, T2) :- release_open(T), causable(open_abstract, T, T2), T2 >= T.

taskTBA(pickup_abstract(OBJ), T) :- achieve_goals_pickup(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ).
taskTBA(release, T2) :- achieve_goals_pickup(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ), causable(pickup_abstract(OBJ), T, T2), T2 >= T.

taskTBA(move_abstract, T) :- achieve_goals_move(T).
taskTBA(achieve_goals, T2) :- achieve_goals_move(T), causable(move_abstract, T, T2), T2 >= T.

taskTBA(open_abstract, T) :- achieve_goals_open(T).
taskTBA(achieve_goals, T2) :- achieve_goals_open(T), causable(open_abstract, T, T2), T2 >= T.


taskTBA(pickup_wrapper(OBJ, LOC), T) :- newmethod22(OBJ, LOC, T), package(OBJ), room(LOC).

taskTBA(putdown_wrapper(OBJ, LOC), T) :- newmethod23(OBJ, LOC, T), package(OBJ), room(LOC).

taskTBA(move_wrapper(LOC1, LOC2, D), T) :- newmethod24(LOC1, LOC2, D, T), room(LOC1), room(LOC2), roomdoor(D).

taskTBA(open_wrapper(LOC1, LOC2, D), T) :- newmethod25(LOC1, LOC2, D, T), room(LOC1), room(LOC2), roomdoor(D).

0 { taskTBA(pickup(OBJ, LOC), T) : m_pickup_wrapper(OBJ, LOC, T), in_state(armempty, T), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), room(LOC), package(OBJ) } 1 :- time(T).

0 { taskTBA(putdown(OBJ, LOC), T) : m_putdown_wrapper(OBJ, LOC, T), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ) } 1 :- time(T).

0 { taskTBA(move(LOC1, LOC2, D), T) : m_move_wrapper(LOC1, LOC2, D, T), in_state(rloc(LOC1), T), in_state(door(LOC1, LOC2, D), T), not in_state(closed(D), T), room(LOC1), roomdoor(D), room(LOC2) } 1 :- time(T).

0 { taskTBA(open(LOC1, LOC2, D), T) : m_open_wrapper(LOC1, LOC2, D, T), in_state(rloc(LOC1), T), in_state(door(LOC1, LOC2, D), T), in_state(closed(D), T), room(LOC1), roomdoor(D), room(LOC2) } 1 :- time(T).

% Causable rules for compound tasks

causable(release, T, T3) :- release_putdown_abstract(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ), causable(putdown_abstract, T, T2), T2 >= T, causable(achieve_goals, T2, T3), T3 >= T2.
causable(release, T, T3) :- release_move(T), causable(move_abstract, T, T2), T2 >= T, causable(release, T2, T3), T3 >= T2.
causable(release, T, T3) :- release_open(T), causable(open_abstract, T, T2), T2 >= T, causable(release, T2, T3), T3 >= T2.

causable(achieve_goals, T, T3) :- achieve_goals_pickup(LOC, OBJ, T), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ), causable(pickup_abstract(OBJ), T, T2), T2 >= T, causable(release, T2, T3), T3 >= T2.
causable(achieve_goals, T, T3) :- achieve_goals_move(T), causable(move_abstract, T, T2), T2 >= T, causable(achieve_goals, T2, T3), T3 >= T2.
causable(achieve_goals, T, T3) :- achieve_goals_open(T), causable(open_abstract, T, T2), T2 >= T, causable(achieve_goals, T2, T3), T3 >= T2.
causable(achieve_goals, T, T) :- finished(T).

causable(pickup_abstract(OBJ), T, T2) :- newmethod22(OBJ, LOC, T), package(OBJ), causable(pickup_wrapper(OBJ, LOC), T, T2), T2 >= T.

causable(putdown_abstract, T, T2) :- newmethod23(OBJ, LOC, T), causable(putdown_wrapper(OBJ, LOC), T, T2), T2 >= T.

causable(move_abstract, T, T2) :- newmethod24(LOC1, LOC2, D, T), causable(move_wrapper(LOC1, LOC2, D), T, T2), T2 >= T.

causable(open_abstract, T, T2) :- newmethod25(LOC1, LOC2, D, T), causable(open_wrapper(LOC1, LOC2, D), T, T2), T2 >= T.

causable(pickup_wrapper(OBJ, LOC), T, T2) :- m_pickup_wrapper(OBJ, LOC, T), in_state(armempty, T), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), room(LOC), package(OBJ), package(OBJ), room(LOC), causable(pickup(OBJ, LOC), T, T2), T2 >= T.

causable(putdown_wrapper(OBJ, LOC), T, T2) :- m_putdown_wrapper(OBJ, LOC, T), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T), room(LOC), package(OBJ), package(OBJ), room(LOC), causable(putdown(OBJ, LOC), T, T2), T2 >= T.

causable(move_wrapper(LOC1, LOC2, D), T, T2) :- m_move_wrapper(LOC1, LOC2, D, T), in_state(rloc(LOC1), T), in_state(door(LOC1, LOC2, D), T), not in_state(closed(D), T), room(LOC1), roomdoor(D), room(LOC2), room(LOC1), room(LOC2), roomdoor(D), causable(move(LOC1, LOC2, D), T, T2), T2 >= T.

causable(open_wrapper(LOC1, LOC2, D), T, T2) :- m_open_wrapper(LOC1, LOC2, D, T), in_state(rloc(LOC1), T), in_state(door(LOC1, LOC2, D), T), in_state(closed(D), T), room(LOC1), roomdoor(D), room(LOC2), room(LOC1), room(LOC2), roomdoor(D), causable(open(LOC1, LOC2, D), T, T2), T2 >= T.
