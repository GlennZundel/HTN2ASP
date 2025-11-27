% Problem-specific facts

% Object declarations
package(o1).
room(c).
room(r1).
room(r2).
room(r3).
roomdoor(d12).
roomdoor(d13).
roomdoor(d01).

% Initial state
in_state(rloc(c), 0).
in_state(armempty, 0).
in_state(door(c, r1, d01), 0).
in_state(door(r1, c, d01), 0).
in_state(door(r1, r2, d12), 0).
in_state(door(r1, r3, d13), 0).
in_state(door(r2, r1, d12), 0).
in_state(door(r3, r1, d13), 0).
in_state(closed(d13), 0).
in_state(in(o1, r3), 0).
in_state(goal_in(o1, r2), 0).

% HTN Tasks
taskTBA(achieve_goals, 0).

% Goal verification
plan_found :- causable(achieve_goals, 0, T1), T1 >= 0, time(T1), in_state(in(o1, r2), T1).