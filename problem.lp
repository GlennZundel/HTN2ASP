#const n=30.
package(o1).
room(c). room(r1). room(r2). room(r3).
roomdoor(d12). roomdoor(d13). roomdoor(d01).
% Initial fluents (dynamic)
holds(rloc(c), 0).
holds(armempty, 0).
holds(closed(d13), 0).
holds(in(o1, r3), 0).

holds(door(c, r1, d01),0).
holds(door(r1, c, d01),0).
holds(door(r1, r2, d12),0).
holds(door(r1, r3, d13),0).
holds(door(r2, r1, d12),0).
holds(door(r3, r1, d13),0).
holds(goal_in(o1, r2),0).

initial_task(achieve_goals).

% Goal constraint: Das Paket muss am Ende im Zielraum sein
:- #count{ T : holds(in(o1, r2), T) } = 0.
