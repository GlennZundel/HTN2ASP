%causable definitions for primitive tasks/ actions
causable(pickup(Obj, Loc), T, T+1) :- taskTBA(pickup(Obj, Loc), T), package(Obj), room(Loc), time(T).
causable(putdown(Obj, Loc), T, T+1) :- taskTBA(putdown(Obj, Loc), T), package(Obj), room(Loc), time(T).
causable(move(Loc1, Loc2, D), T, T+1) :- taskTBA(move(Loc1, Loc2, D), T), room(Loc1), room(Loc2), roomdoor(D), time(T).
causable(open(Loc1, Loc2, D), T, T+1) :- taskTBA(open(Loc1, Loc2, D), T), room(Loc1), room(Loc2), roomdoor(D), time(T).

%operators for primitve tasks/ actions
out_state(in(Obj, Loc), T+1) :- taskTBA(pickup(Obj, Loc), T), package(Obj), room(Loc), time(T).
out_state(armempty, T+1) :- taskTBA(pickup(Obj, Loc), T), package(Obj), room(Loc), time(T).
in_state(holding(Obj), T+1) :- taskTBA(pickup(Obj, Loc), T), package(Obj), room(Loc), time(T).

out_state(holding(Obj), T+1) :- taskTBA(putdown(Obj, Loc), T), package(Obj), room(Loc), time(T).
in_state(armempty, T+1) :- taskTBA(putdown(Obj, Loc), T), package(Obj), room(Loc), time(T).
in_state(in(Obj, Loc), T+1) :- taskTBA(putdown(Obj, Loc), T), package(Obj), room(Loc), time(T).

in_state(rloc(Loc2), T+1) :- taskTBA(move(Loc1, Loc2, D), T), room(Loc1), room(Loc2), roomdoor(D), time(T).
out_state(rloc(Loc1), T+1) :- taskTBA(move(Loc1, Loc2, D), T), room(Loc1), room(Loc2), roomdoor(D), time(T).

out_state(closed(D), T+1) :- taskTBA(open(Loc1, Loc2, D), T), room(Loc1), room(Loc2), roomdoor(D), time(T).

%always choose only one method:
achieve_goals_pickup(T) :- taskTBA(achieve_goals, T), not  achieve_goals_move(T), not  achieve_goals_open(T), not  finished(T), time(T).
 achieve_goals_move(T) :- taskTBA(achieve_goals, T), not  achieve_goals_pickup(T), not  achieve_goals_open(T), not  finished(T), time(T).
 achieve_goals_open(T) :- taskTBA(achieve_goals, T), not  achieve_goals_pickup(T), not  achieve_goals_move(T), not  finished(T), time(T).
 finished(T) :- taskTBA(achieve_goals, T), not  achieve_goals_pickup(T), not  achieve_goals_move(T), not  achieve_goals_open(T), time(T).

release_putdown_abstract(T) :- taskTBA(release, T), not  release_move(T), not  release_open(T), time(T).
 release_move(T) :- taskTBA(release, T), not  release_putdown_abstract(T), not  release_open(T), time(T).
 release_open(T) :- taskTBA(release, T), not  release_putdown_abstract(T), not  release_move(T), time(T).

newMethod22(T) :- taskTBA(pickup_abstract(Obj), T), package(Obj), time(T).

newMethod23(T) :- taskTBA(putdown_abstract, T), time(T).

newMethod24(T) :- taskTBA(move_abstract, T), time(T).

newMethod25(T) :- taskTBA(open_abstract, T), time(T).


%preconditions of methods
checked_state(rloc(Loc), T) :- in_state(rloc(Loc), T), release_putdown_abstract(T), time(T), room(Loc).
checked_state(holding(Obj), T) :- in_state(holding(Obj), T), release_putdown_abstract(T), time(T), package(Obj).
checked_state(goal_in(Obj, Loc), T) :- in_state(goal_in(Obj, Loc), T), release_putdown_abstract(T), time(T), package(Obj), room(Loc).

checked_state(rloc(Loc), T) :- in_state(rloc(Loc), T), achieve_goals_pickup(T), time(T), room(Loc).
checked_state(in(Obj, Loc), T) :- in_state(in(Obj, Loc), T), achieve_goals_pickup(T), time(T), package(Obj), room(Loc).
checked_state(goal_in(Obj, Loc), T) :- not in_state(goal_in(Obj, Loc), T), achieve_goals_pickup(T), time(T), room(Loc), package(Obj).

checked_state(armempty, T) :- in_state(armempty, T), newMethod22(T), time(T).
checked_state(rloc(Loc), T) :- in_state(rloc(Loc), T), newMethod22(T), time(T), room(Loc).
checked_state(in(Obj, Loc), T) :- in_state(in(Obj, Loc), T), newMethod22(T), time(T), package(Obj), room(Loc).

checked_state(rloc(Loc), T) :- in_state(rloc(Loc), T), newMethod23(T), time(T), room(Loc).
checked_state(holding(Obj), T) :- in_state(holding(Obj), T), newMethod23(T), time(T), package(Obj).
checked_state(goal_in(Obj, Loc), T) :- in_state(goal_in(Obj, Loc), T), newMethod23(T), time(T), package(Obj), room(Loc).

checked_state(rloc(Loc1), T) :- in_state(rloc(Loc1), T), newMethod24(T), time(T), room(Loc1).
checked_state(door(Loc1, Loc2, D), T) :- in_state(door(Loc1, Loc2, D), T), newMethod24(T), time(T), room(Loc1), room(Loc2), roomdoor(D).
checked_state(not_closed(D), T) :- not in_state(closed(D), T), newMethod24(T), time(T), roomdoor(D).

checked_state(rloc(Loc1), T) :- in_state(rloc(Loc1), T), newMethod25(T), time(T), room(Loc1).
checked_state(door(Loc1, Loc2, D), T) :- in_state(door(Loc1, Loc2, D), T), newMethod25(T), time(T), room(Loc1), room(Loc2), roomdoor(D).
checked_state(closed(D), T) :- in_state(closed(D), T), newMethod25(T), time(T), roomdoor(D).

%decomposition of methods
taskTBA(putdown_abstract, T) :- release_putdown_abstract(T), checked_state(rloc(Loc), T), checked_state(holding(Obj), T), checked_state(goal_in(Obj, Loc), T), room(Loc), package(Obj).
taskTBA(achieve_goals, T2) :- release_putdown_abstract(T), checked_state(rloc(Loc), T), checked_state(holding(Obj), T), checked_state(goal_in(Obj, Loc), T), causable(putdown_abstract, T, T2), T2 >= T, room(Loc), package(Obj), time(T), time(T2).

taskTBA(move_abstract, T) :- release_move(T), time(T).
taskTBA(release, T2) :- release_move(T), causable(move_abstract, T, T2), T2 >= T, time(T), time(T2).

taskTBA(open_abstract, T) :- release_open(T), time(T).
taskTBA(release, T2) :- release_open(T), causable(open_abstract, T, T2), T2 >= T, time(T), time(T2).

taskTBA(pickup_abstract(Obj), T) :- achieve_goals_pickup(T), checked_state(rloc(Loc), T), checked_state(in(Obj, Loc), T), checked_state(goal_in(Obj, Loc), T), room(Loc), package(Obj), time(T).
taskTBA(release, T2) :- achieve_goals_pickup(T), checked_state(rloc(Loc), T), checked_state(in(Obj, Loc), T), checked_state(goal_in(Obj, Loc), T), causable(pickup_abstract(Obj), T, T2), T2 >= T, room(Loc), package(Obj), time(T), time(T2).

taskTBA(move_abstract, T) :- achieve_goals_move(T), time(T).
taskTBA(achieve_goals, T2) :- achieve_goals_move(T), causable(move_abstract, T, T2), T2 >= T, time(T), time(T2).

taskTBA(open_abstract, T) :- achieve_goals_open(T), time(T).
taskTBA(achieve_goals, T2) :- achieve_goals_open(T), causable(open_abstract, T, T2), T2 >= T, time(T), time(T2).

0 { taskTBA(pickup(Obj, Loc), T) : newMethod22(T), checked_state(armempty, T), checked_state(rloc(Loc), T), checked_state(in(Obj, Loc), T), room(Loc), package(Obj) } 1 :- time(T).

0 { taskTBA(putdown(Obj, Loc), T) : newMethod23(T), checked_state(rloc(Loc), T), checked_state(holding(Obj), T), checked_state(goal_in(Obj, Loc), T), room(Loc), package(Obj) } 1 :- time(T).

0 { taskTBA(move(Loc1, Loc2, D), T) : newMethod24(T), checked_state(rloc(Loc1), T), checked_state(door(Loc1, Loc2, D), T), checked_state(not_closed(D), T), room(Loc1), room(Loc2), roomdoor(D) } 1 :- time(T).

0 { taskTBA(open(Loc1, Loc2, D), T) : newMethod25(T), checked_state(rloc(Loc1), T), checked_state(door(Loc1, Loc2, D), T), checked_state(closed(D), T), room(Loc1), room(Loc2), roomdoor(D) } 1 :- time(T).

%accomplishment of compund tasks
causable(achieve_goals, T, T3) :- achieve_goals_pickup(T), checked_state(rloc(Loc), T), checked_state(in(Obj, Loc), T), checked_state(goal_in(Obj, Loc), T), causable(pickup_abstract(Obj), T, T2), causable(release, T2, T3), T3 >= T2, T2 >= T, room(Loc), package(Obj), time(T), time(T3), time(T2).
causable(achieve_goals, T, T3) :- achieve_goals_move(T), causable(move_abstract, T, T2), causable(achieve_goals, T2, T3), T3 >= T2, T2 >= T, time(T), time(T2), time(T3).
causable(achieve_goals, T, T3) :- achieve_goals_open(T), causable(open_abstract, T, T2), causable(achieve_goals, T2, T3), T3 >= T2, T2 >= T, time(T), time(T2), time(T3).
causable(achieve_goals, T, T) :- finished(T).

causable(release, T, T3) :- release_putdown_abstract(T), checked_state(rloc(Loc), T), checked_state(holding(Obj), T), checked_state(goal_in(Obj, Loc), T), causable(putdown_abstract, T, T2), causable(achieve_goals, T2, T3), T3 >= T2, T2 >= T, room(Loc), package(Obj), time(T), time(T2), time(T3).
causable(release, T, T3) :- release_move(T), causable(move_abstract, T, T2), causable(release, T2, T3), T3 >= T2, T2 >= T, time(T), time(T2), time(T3).
causable(release, T, T3) :- release_open(T), causable(open_abstract, T, T2), causable(release, T2, T3), T3 >= T2, T2 >= T, time(T), time(T2), time(T3).

causable(pickup_abstract(Obj), T, T2) :- newMethod22(T), checked_state(armempty, T), checked_state(rloc(Loc), T), checked_state(in(Obj, Loc), T), causable(pickup(Obj, Loc), T, T2), T2 >= T, room(Loc), package(Obj), time(T), time(T2).

causable(putdown_abstract, T, T2) :- newMethod23(T), checked_state(rloc(Loc), T), checked_state(holding(Obj), T), checked_state(goal_in(Obj, Loc), T), causable(putdown(Obj, Loc), T, T2), T2 >= T, room(Loc), package(Obj), time(T), time(T2).

causable(move_abstract, T, T2) :- newMethod24(T), checked_state(rloc(Loc1), T), checked_state(door(Loc1, Loc2, D), T), checked_state(not_closed(D), T), causable(move(Loc1, Loc2, D), T, T2), T2 >= T, room(Loc1), room(Loc2), roomdoor(D), time(T), time(T2).

causable(open_abstract, T, T2) :- newMethod25(T), checked_state(rloc(Loc1), T), checked_state(door(Loc1, Loc2, D), T), checked_state(closed(D), T), causable(open(Loc1, Loc2, D), T, T2), T2 >= T, room(Loc1), room(Loc2), roomdoor(D), time(T), time(T2).