% Problem-specific facts

% Object declarations
package(o1).
package(o2).
package(o3).
package(o4).
package(o5).
room(c).
room(r1).
room(r2).
room(r3).
room(r4).
room(r5).
roomdoor(d01).
roomdoor(d04).
roomdoor(d02).
roomdoor(d25).
roomdoor(d23).

% Initial state
in_state(rloc(c), 0).
in_state(armempty, 0).
in_state(door(c, r1, d01), 0).
in_state(door(c, r2, d02), 0).
in_state(door(c, r4, d04), 0).
in_state(door(r1, c, d01), 0).
in_state(door(r2, c, d02), 0).
in_state(door(r2, r3, d23), 0).
in_state(door(r2, r5, d25), 0).
in_state(door(r3, r2, d23), 0).
in_state(door(r4, c, d04), 0).
in_state(door(r5, r2, d25), 0).
in_state(closed(d04), 0).
in_state(closed(d02), 0).
in_state(in(o1, r5), 0).
in_state(in(o2, r4), 0).
in_state(in(o3, r4), 0).
in_state(in(o4, r3), 0).
in_state(in(o5, r1), 0).
in_state(goal_in(o1, r4), 0).
in_state(goal_in(o2, r3), 0).
in_state(goal_in(o3, r1), 0).
in_state(goal_in(o4, r5), 0).
in_state(goal_in(o5, r5), 0).

% HTN Tasks
taskTBA(achieve_goals, 0).

% Goal verification
plan_found :- causable(achieve_goals, 0, T1), T1 >= 0, time(T1), in_state(in(o1, r4), T1), in_state(in(o2, r3), T1), in_state(in(o3, r1), T1), in_state(in(o4, r5), T1), in_state(in(o5, r5), T1).