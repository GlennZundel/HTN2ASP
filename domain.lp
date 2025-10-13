time(0..n).


%%compound tasks
abstract(achieve_goals).
abstract(release).
abstract(pickup_abstract(Obj, Loc)) :- package(Obj), room(Loc).
abstract(putdown_abstract(Obj, Loc)) :- package(Obj), room(Loc).
abstract(move_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
abstract(open_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

%%primitive tasks
action(pickup(Obj, Loc)) :- package(Obj), room(Loc).
primitive(pickup(Obj, Loc)) :- package(Obj), room(Loc).
precondition(pickup(Obj, Loc), armempty) :- package(Obj), room(Loc).
precondition(pickup(Obj, Loc), rloc(Loc)) :- package(Obj), room(Loc).
precondition(pickup(Obj, Loc), in(Obj, Loc)) :- package(Obj), room(Loc).
adds(pickup(Obj, Loc), holding(Obj)) :- package(Obj), room(Loc).
deletes(pickup(Obj, Loc), in(Obj, Loc)) :- package(Obj), room(Loc).
deletes(pickup(Obj, Loc), armempty) :- package(Obj), room(Loc).

action(putdown(Obj, Loc)) :- package(Obj), room(Loc).
primitive(putdown(Obj, Loc)) :- package(Obj), room(Loc).
precondition(putdown(Obj, Loc), rloc(Loc)) :- package(Obj), room(Loc).
precondition(putdown(Obj, Loc), holding(Obj)) :- package(Obj), room(Loc).
precondition(putdown(Obj, Loc), goal_in(Obj, Loc)) :- package(Obj), room(Loc).
adds(putdown(Obj, Loc), armempty) :- package(Obj), room(Loc).
adds(putdown(Obj, Loc), in(Obj, Loc)) :- package(Obj), room(Loc).
deletes(putdown(Obj, Loc), holding(Obj)) :- package(Obj), room(Loc).

action(move(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
primitive(move(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
precondition(move(Loc1, Loc2, D), rloc(Loc1)) :- room(Loc1), room(Loc2), roomdoor(D).
precondition(move(Loc1, Loc2, D), door(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
neg_precondition(move(Loc1, Loc2, D), closed(D)) :- room(Loc1), room(Loc2), roomdoor(D).
adds(move(Loc1, Loc2, D), rloc(Loc2)) :- room(Loc1), room(Loc2), roomdoor(D).
deletes(move(Loc1, Loc2, D), rloc(Loc1)) :- room(Loc1), room(Loc2), roomdoor(D).

action(open(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
primitive(open(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
precondition(open(Loc1, Loc2, D), rloc(Loc1)) :- room(Loc1), room(Loc2), roomdoor(D).
precondition(open(Loc1, Loc2, D), door(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
precondition(open(Loc1, Loc2, D), closed(D)) :- room(Loc1), room(Loc2), roomdoor(D).
deletes(open(Loc1, Loc2, D), closed(D)) :- room(Loc1), room(Loc2), roomdoor(D).


%%methods
decomposes(release_putdown_abstract(Loc, Obj), release) :- room(Loc), package(Obj).
method_precondition(release_putdown_abstract(Loc, Obj), rloc(Loc)) :- room(Loc), package(Obj).
method_precondition(release_putdown_abstract(Loc, Obj), holding(Obj)):- room(Loc), package(Obj).
method_precondition(release_putdown_abstract(Loc, Obj), goal_in(Obj, Loc)) :- room(Loc), package(Obj).
subtask(release_putdown_abstract(Loc, Obj), 1, putdown_abstract(Obj, Loc)) :- room(Loc), package(Obj).
subtask(release_putdown_abstract(Loc, Obj), 2, achieve_goals) :- room(Loc), package(Obj).
method(release_putdown_abstract(Loc, Obj)) :- room(Loc), package(Obj).

decomposes(release_move(Loc1, Loc2, D), release) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(release_move(Loc1, Loc2, D), 1, move_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(release_move(Loc1, Loc2, D), 2, release) :- room(Loc1), room(Loc2), roomdoor(D).
method(release_move(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

decomposes(release_open(Loc1, Loc2, D), release) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(release_open(Loc1, Loc2, D), 1, open_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(release_open(Loc1, Loc2, D), 2, release) :- room(Loc1), room(Loc2), roomdoor(D).
method(release_open(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

decomposes(achieve_goals_pickup(Loc, Obj), achieve_goals) :- room(Loc), package(Obj).
method_precondition(achieve_goals_pickup(Loc, Obj), rloc(Loc)) :- room(Loc), package(Obj).
method_precondition(achieve_goals_pickup(Loc, Obj), in(Obj, Loc)) :- room(Loc), package(Obj).
neg_method_precondition(achieve_goals_pickup(Loc, Obj), goal_in(Obj, Loc)) :- room(Loc), package(Obj).
subtask(achieve_goals_pickup(Loc, Obj), 1, pickup_abstract(Obj, Loc)) :- room(Loc), package(Obj).
subtask(achieve_goals_pickup(Loc, Obj), 2, release) :- room(Loc), package(Obj).
method(achieve_goals_pickup(Loc, Obj)) :- room(Loc), package(Obj).

decomposes(achieve_goals_move(Loc1, Loc2, D), achieve_goals) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(achieve_goals_move(Loc1, Loc2, D), 1, move_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(achieve_goals_move(Loc1, Loc2, D), 2, achieve_goals) :- room(Loc1), room(Loc2), roomdoor(D).
method(achieve_goals_move(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

decomposes(achieve_goals_open(Loc1, Loc2, D), achieve_goals) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(achieve_goals_open(Loc1, Loc2, D), 1, open_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(achieve_goals_open(Loc1, Loc2, D), 2, achieve_goals) :- room(Loc1), room(Loc2), roomdoor(D).
method(achieve_goals_open(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

decomposes(finished, achieve_goals).
method_precondition(finished, in(o1, r2)).
method(finished).

decomposes(newMethod22(Obj, Loc), pickup_abstract(Obj, Loc)) :- room(Loc), package(Obj).
subtask(newMethod22(Obj, Loc), 1, pickup(Obj, Loc)) :- room(Loc), package(Obj).
method(newMethod22(Obj, Loc)) :- package(Obj), room(Loc).

decomposes(newMethod23(Obj, Loc), putdown_abstract(Obj, Loc)) :- room(Loc), package(Obj).
subtask(newMethod23(Obj, Loc), 1, putdown(Obj, Loc)) :- room(Loc), package(Obj).
method(newMethod23(Obj, Loc)) :- package(Obj), room(Loc).

decomposes(newMethod24(Loc1, Loc2, D), move_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(newMethod24(Loc1, Loc2, D), 1, move(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
method(newMethod24(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).

decomposes(newMethod25(Loc1, Loc2, D), open_abstract(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
subtask(newMethod25(Loc1, Loc2, D), 1, open(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).
method(newMethod25(Loc1, Loc2, D)) :- room(Loc1), room(Loc2), roomdoor(D).