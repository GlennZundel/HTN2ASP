#const n=20.
package(o1).
room(c). room(r1). room(r2). room(r3).
roomdoor(d12). roomdoor(d13). roomdoor(d01).
% Initial fluents (dynamic)
holds(rloc(c), 0).
holds(armempty, 0).
holds(closed(d13), 0).
holds(in(o1, r3), 0).

% Static predicates (never change)
door(c, r1, d01).
door(r1, c, d01).
door(r1, r2, d12).
door(r1, r3, d13).
door(r2, r1, d12).
door(r3, r1, d13).
goal_in(o1, r2).

initial_task(achieve_goals).

% Goal constraint: Das Paket muss am Ende im Zielraum sein
:- not holds(in(o1, r2), n).

