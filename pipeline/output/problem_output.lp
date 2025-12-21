% Object declarations
depot(depot0).
distributor(distributor0).
distributor(distributor1).
truck(truck0).
truck(truck1).
pallet(pallet0).
pallet(pallet1).
pallet(pallet2).
crate(crate0).
crate(crate1).
crate(crate2).
crate(crate3).
crate(crate4).
crate(crate5).
crate(crate6).
crate(crate7).
hoist(hoist0).
hoist(hoist1).
hoist(hoist2).

% Initial state
in_state(at(pallet0, depot0), 0).
in_state(clear(crate7), 0).
in_state(at(pallet1, distributor0), 0).
in_state(clear(crate2), 0).
in_state(at(pallet2, distributor1), 0).
in_state(clear(crate6), 0).
in_state(at(truck0, distributor1), 0).
in_state(at(truck1, distributor1), 0).
in_state(at(hoist0, depot0), 0).
in_state(available(hoist0), 0).
in_state(at(hoist1, distributor0), 0).
in_state(available(hoist1), 0).
in_state(at(hoist2, distributor1), 0).
in_state(available(hoist2), 0).
in_state(at(crate0, depot0), 0).
in_state(on(crate0, pallet0), 0).
in_state(at(crate1, depot0), 0).
in_state(on(crate1, crate0), 0).
in_state(at(crate2, distributor0), 0).
in_state(on(crate2, pallet1), 0).
in_state(at(crate3, distributor1), 0).
in_state(on(crate3, pallet2), 0).
in_state(at(crate4, depot0), 0).
in_state(on(crate4, crate1), 0).
in_state(at(crate5, distributor1), 0).
in_state(on(crate5, crate3), 0).
in_state(at(crate6, distributor1), 0).
in_state(on(crate6, crate5), 0).
in_state(at(crate7, depot0), 0).
in_state(on(crate7, crate4), 0).

% HTN Tasks
taskTBA(do_put_on(crate7, pallet0), 0).
taskTBA(do_put_on(crate6, pallet1), TT) :- causable(do_put_on(crate7, pallet0), 0, TT), TT >= 0, time(TT).
taskTBA(do_put_on(crate5, pallet2), TT2) :- causable(do_put_on(crate6, pallet1), TT, TT2), TT2 >= TT, time(TT), time(TT2).
taskTBA(do_put_on(crate4, crate7), TT3) :- causable(do_put_on(crate5, pallet2), TT2, TT3), TT3 >= TT2, time(TT2), time(TT3).
taskTBA(do_put_on(crate2, crate6), TT4) :- causable(do_put_on(crate4, crate7), TT3, TT4), TT4 >= TT3, time(TT3), time(TT4).
taskTBA(do_put_on(crate0, crate4), TT5) :- causable(do_put_on(crate2, crate6), TT4, TT5), TT5 >= TT4, time(TT4), time(TT5).

% Goal verification
plan_found :- causable(do_put_on(crate0, crate4), TT5, TT6), TT6 >= TT5, time(TT6), time(TT5), in_state(on(crate0, crate4), TT6), in_state(on(crate2, crate6), TT6), in_state(on(crate4, crate7), TT6), in_state(on(crate5, pallet2), TT6), in_state(on(crate6, pallet1), TT6), in_state(on(crate7, pallet0), TT6).