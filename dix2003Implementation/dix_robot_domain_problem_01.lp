room(c).
room(r1).
room(r2).
room(r3).
package(o1).
roomdoor(d01).
roomdoor(d12).
roomdoor(d13).

in_state(rloc(c), 0).
in_state(armempty, 0).
in_state(door(c, r1, d01), 0).
in_state(door(r1, c, d01), 0).
in_state(door(r1, r2, d12), 0).
in_state(door(r2, r1, d12), 0).
in_state(door(r1, r3, d13), 0).
in_state(door(r3, r1, d13), 0).
in_state(closed(d13), 0).
in_state(in(o1, r3), 0).
in_state(goal_in(o1, r2), 0).

taskTBA(achieve_goals, 0).

plan_found :- causable(achieve_goals, T1, T2), T2 >= T1, in_state(in(o1, r2), T2), time(T1), time(T2). %%