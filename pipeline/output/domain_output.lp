% Primitive tasks (actions)

% Type hierarchy
place(XX) :- depot(XX).
place(XX) :- distributor(XX).
locatable(XX) :- truck(XX).
locatable(XX) :- hoist(XX).
locatable(XX) :- surface(XX).
surface(XX) :- pallet(XX).
surface(XX) :- crate(XX).

% Action declarations
action(drive(X, Y, Z)) :- truck(X), place(Y), place(Z).
action(drop(X, Y, Z, P)) :- hoist(X), crate(Y), surface(Z), place(P).
action(lift(X, Y, Z, P)) :- hoist(X), crate(Y), surface(Z), place(P).
action(load(X, Y, Z, P)) :- hoist(X), crate(Y), truck(Z), place(P).
action(nop).
action(unload(X, Y, Z, P)) :- hoist(X), crate(Y), truck(Z), place(P).

causable(drive(X, Y, Z), TT, TT+1) :- taskTBA(drive(X, Y, Z), TT), truck(X), place(Y), place(Z).
out_state(at(X, Y), TT+1) :- taskTBA(drive(X, Y, Z), TT).
in_state(at(X, Z), TT+1) :- taskTBA(drive(X, Y, Z), TT).

causable(lift(X, Y, Z, P), TT, TT+1) :- taskTBA(lift(X, Y, Z, P), TT), hoist(X), crate(Y), surface(Z), place(P).
out_state(at(Y, P), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).
in_state(lifting(X, Y), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).
out_state(clear(Y), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).
out_state(available(X), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).
in_state(clear(Z), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).
out_state(on(Y, Z), TT+1) :- taskTBA(lift(X, Y, Z, P), TT).

causable(drop(X, Y, Z, P), TT, TT+1) :- taskTBA(drop(X, Y, Z, P), TT), hoist(X), crate(Y), surface(Z), place(P).
in_state(available(X), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).
out_state(lifting(X, Y), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).
in_state(at(Y, P), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).
out_state(clear(Z), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).
in_state(clear(Y), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).
in_state(on(Y, Z), TT+1) :- taskTBA(drop(X, Y, Z, P), TT).

causable(load(X, Y, Z, P), TT, TT+1) :- taskTBA(load(X, Y, Z, P), TT), hoist(X), crate(Y), truck(Z), place(P).
out_state(lifting(X, Y), TT+1) :- taskTBA(load(X, Y, Z, P), TT).
in_state(in(Y, Z), TT+1) :- taskTBA(load(X, Y, Z, P), TT).
in_state(available(X), TT+1) :- taskTBA(load(X, Y, Z, P), TT).

causable(unload(X, Y, Z, P), TT, TT+1) :- taskTBA(unload(X, Y, Z, P), TT), hoist(X), crate(Y), truck(Z), place(P).
out_state(in(Y, Z), TT+1) :- taskTBA(unload(X, Y, Z, P), TT).
out_state(available(X), TT+1) :- taskTBA(unload(X, Y, Z, P), TT).
in_state(lifting(X, Y), TT+1) :- taskTBA(unload(X, Y, Z, P), TT).

causable(nop, TT, TT+1) :- taskTBA(nop, TT).

% Method selection rules

m0_do_put_on(do_put_on, C, S2, TT) :- taskTBA(do_put_on(C, S2), TT), crate(C), surface(S2), not m1_do_put_on(do_put_on, C, S2, P, H, TT), crate(C), surface(S2), place(P), hoist(H), not m2_do_put_on(do_put_on, C, S2, P, T, H, TT), crate(C), surface(S2), place(P), truck(T), hoist(H), not m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), crate(C), surface(S2), surface(S1), place(P1), truck(T), place(P2).
m1_do_put_on(do_put_on, C, S2, P, H, TT) :- taskTBA(do_put_on(C, S2), TT), crate(C), hoist(H), place(P), surface(S2), not m0_do_put_on(do_put_on, C, S2, TT), crate(C), surface(S2), not m2_do_put_on(do_put_on, C, S2, P, T, H, TT), crate(C), surface(S2), place(P), truck(T), hoist(H), not m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), crate(C), surface(S2), surface(S1), place(P1), truck(T), place(P2).
m2_do_put_on(do_put_on, C, S2, P, T, H, TT) :- taskTBA(do_put_on(C, S2), TT), crate(C), hoist(H), place(P), surface(S2), truck(T), not m0_do_put_on(do_put_on, C, S2, TT), crate(C), surface(S2), not m1_do_put_on(do_put_on, C, S2, P, H, TT), crate(C), surface(S2), place(P), hoist(H), not m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), crate(C), surface(S2), surface(S1), place(P1), truck(T), place(P2).
m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT) :- taskTBA(do_put_on(C, S2), TT), crate(C), place(P1), place(P2), surface(S1), surface(S2), truck(T), not m0_do_put_on(do_put_on, C, S2, TT), crate(C), surface(S2), not m1_do_put_on(do_put_on, C, S2, P, H, TT), crate(C), surface(S2), place(P), hoist(H), not m2_do_put_on(do_put_on, C, S2, P, T, H, TT), crate(C), surface(S2), place(P), truck(T), hoist(H).

m4_do_clear(do_clear, S1, P1, TT) :- taskTBA(do_clear(S1, P1), TT), place(P1), surface(S1), not m5_do_clear(do_clear, S1, P1, C, T, H1, TT), surface(S1), place(P1), crate(C), truck(T), hoist(H1).
m5_do_clear(do_clear, S1, P1, C, T, H1, TT) :- taskTBA(do_clear(S1, P1), TT), crate(C), hoist(H1), place(P1), surface(S1), truck(T), not m4_do_clear(do_clear, S1, P1, TT), surface(S1), place(P1).

m6_do_get_truck(do_get_truck, T, P1, TT) :- taskTBA(do_get_truck(T, P1), TT), place(P1), truck(T), not m7_do_get_truck(do_get_truck, T, P1, P2, TT), truck(T), place(P1), place(P2).
m7_do_get_truck(do_get_truck, T, P1, P2, TT) :- taskTBA(do_get_truck(T, P1), TT), place(P1), place(P2), truck(T), not m6_do_get_truck(do_get_truck, T, P1, TT), truck(T), place(P1).

m8_do_lift_crate(do_lift_crate, C, P, H, T, TT) :- taskTBA(do_lift_crate(C, P, H), TT), crate(C), hoist(H), place(P), truck(T), not m9_do_lift_crate(do_lift_crate, C, P, H, S, TT), crate(C), place(P), hoist(H), surface(S).
m9_do_lift_crate(do_lift_crate, C, P, H, S, TT) :- taskTBA(do_lift_crate(C, P, H), TT), crate(C), hoist(H), place(P), surface(S), not m8_do_lift_crate(do_lift_crate, C, P, H, T, TT), crate(C), place(P), hoist(H), truck(T).

m10_do_load_truck(do_load_truck, C, S, P, T, H, TT) :- taskTBA(do_load_truck(C, S, P, T), TT), crate(C), hoist(H), place(P), surface(S), truck(T).

m11_do_unload_truck(do_unload_truck, C, S, P, T, H, TT) :- taskTBA(do_unload_truck(C, S, P, T), TT), crate(C), hoist(H), place(P), surface(S), truck(T).

m_drive_wrapper(drive_wrapper, X, Y, Z, TT) :- taskTBA(drive_wrapper(X, Y, Z), TT), place(Y), place(Z), truck(X).

m_lift_wrapper(lift_wrapper, X, Y, Z, P, TT) :- taskTBA(lift_wrapper(X, Y, Z, P), TT), crate(Y), hoist(X), place(P), surface(Z).

m_drop_wrapper(drop_wrapper, X, Y, Z, P, TT) :- taskTBA(drop_wrapper(X, Y, Z, P), TT), crate(Y), hoist(X), place(P), surface(Z).

m_load_wrapper(load_wrapper, X, Y, Z, P, TT) :- taskTBA(load_wrapper(X, Y, Z, P), TT), crate(Y), hoist(X), place(P), truck(Z).

m_unload_wrapper(unload_wrapper, X, Y, Z, P, TT) :- taskTBA(unload_wrapper(X, Y, Z, P), TT), crate(Y), hoist(X), place(P), truck(Z).

% Subtask rules

0 { taskTBA(nop, TT) : m0_do_put_on(do_put_on, C, S2, TT), in_state(on(C, S2), TT), surface(S2), crate(C) } 1 :- time(TT).

taskTBA(do_clear(C, P), TT) :- m1_do_put_on(do_put_on, C, S2, P, H, TT), in_state(at(C, P), TT), crate(C), place(P), surface(C).
taskTBA(do_clear(S2, P), TT2) :- m1_do_put_on(do_put_on, C, S2, P, H, TT), in_state(at(C, P), TT), crate(C), place(P), causable(do_clear(C, P), TT, TT2), TT2 >= TT, surface(S2).
taskTBA(do_lift_crate(C, P, H), TT3) :- m1_do_put_on(do_put_on, C, S2, P, H, TT), in_state(at(C, P), TT), crate(C), place(P), causable(do_clear(S2, P), TT2, TT3), TT3 >= TT2, hoist(H).
taskTBA(drop_wrapper(H, C, S2, P), TT4) :- m1_do_put_on(do_put_on, C, S2, P, H, TT), in_state(at(C, P), TT), crate(C), place(P), causable(do_lift_crate(C, P, H), TT3, TT4), TT4 >= TT3, hoist(H), surface(S2).

taskTBA(do_get_truck(T, P), TT) :- m2_do_put_on(do_put_on, C, S2, P, T, H, TT), in_state(in(C, T), TT), truck(T), crate(C), place(P).
taskTBA(do_clear(S2, P), TT2) :- m2_do_put_on(do_put_on, C, S2, P, T, H, TT), in_state(in(C, T), TT), truck(T), crate(C), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT, surface(S2), place(P).
taskTBA(unload_wrapper(H, C, T, P), TT3) :- m2_do_put_on(do_put_on, C, S2, P, T, H, TT), in_state(in(C, T), TT), truck(T), crate(C), causable(do_clear(S2, P), TT2, TT3), TT3 >= TT2, hoist(H), place(P).
taskTBA(drop_wrapper(H, C, S2, P), TT4) :- m2_do_put_on(do_put_on, C, S2, P, T, H, TT), in_state(in(C, T), TT), truck(T), crate(C), causable(unload_wrapper(H, C, T, P), TT3, TT4), TT4 >= TT3, hoist(H), surface(S2), place(P).

taskTBA(do_load_truck(C, S1, P1, T), TT) :- m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), crate(C), surface(S1), place(P1), truck(T).
taskTBA(drive_wrapper(T, P1, P2), TT2) :- m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), causable(do_load_truck(C, S1, P1, T), TT, TT2), TT2 >= TT, truck(T), place(P1), place(P2).
taskTBA(do_unload_truck(C, S2, P2, T), TT3) :- m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), causable(drive_wrapper(T, P1, P2), TT2, TT3), TT3 >= TT2, crate(C), surface(S2), place(P2), truck(T).

0 { taskTBA(nop, TT) : m4_do_clear(do_clear, S1, P1, TT), in_state(clear(S1), TT), in_state(at(S1, P1), TT), surface(S1), place(P1) } 1 :- time(TT).

taskTBA(do_clear(C, P1), TT) :- m5_do_clear(do_clear, S1, P1, C, T, H1, TT), not in_state(clear(S1), TT), in_state(on(C, S1), TT), in_state(at(S1, P1), TT), in_state(at(H1, P1), TT), hoist(H1), surface(S1), crate(C), place(P1), surface(C).
taskTBA(lift_wrapper(H1, C, S1, P1), TT2) :- m5_do_clear(do_clear, S1, P1, C, T, H1, TT), not in_state(clear(S1), TT), in_state(on(C, S1), TT), in_state(at(S1, P1), TT), in_state(at(H1, P1), TT), hoist(H1), surface(S1), crate(C), place(P1), causable(do_clear(C, P1), TT, TT2), TT2 >= TT.
taskTBA(do_get_truck(T, P1), TT3) :- m5_do_clear(do_clear, S1, P1, C, T, H1, TT), not in_state(clear(S1), TT), in_state(on(C, S1), TT), in_state(at(S1, P1), TT), in_state(at(H1, P1), TT), hoist(H1), surface(S1), crate(C), place(P1), causable(lift_wrapper(H1, C, S1, P1), TT2, TT3), TT3 >= TT2, truck(T).
taskTBA(load_wrapper(H1, C, T, P1), TT4) :- m5_do_clear(do_clear, S1, P1, C, T, H1, TT), not in_state(clear(S1), TT), in_state(on(C, S1), TT), in_state(at(S1, P1), TT), in_state(at(H1, P1), TT), hoist(H1), surface(S1), crate(C), place(P1), causable(do_get_truck(T, P1), TT3, TT4), TT4 >= TT3, truck(T).

0 { taskTBA(nop, TT) : m6_do_get_truck(do_get_truck, T, P1, TT), in_state(at(T, P1), TT), truck(T), place(P1) } 1 :- time(TT).

taskTBA(drive_wrapper(T, P2, P1), TT) :- m7_do_get_truck(do_get_truck, T, P1, P2, TT), not in_state(at(T, P1), TT), truck(T), place(P1), place(P2).

taskTBA(do_get_truck(T, P), TT) :- m8_do_lift_crate(do_lift_crate, C, P, H, T, TT), in_state(in(C, T), TT), in_state(at(H, P), TT), truck(T), place(P), crate(C), hoist(H).
taskTBA(unload_wrapper(H, C, T, P), TT2) :- m8_do_lift_crate(do_lift_crate, C, P, H, T, TT), in_state(in(C, T), TT), in_state(at(H, P), TT), truck(T), place(P), crate(C), hoist(H), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT.

taskTBA(lift_wrapper(H, C, S, P), TT) :- m9_do_lift_crate(do_lift_crate, C, P, H, S, TT), in_state(on(C, S), TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S).

taskTBA(do_get_truck(T, P), TT) :- m10_do_load_truck(do_load_truck, C, S, P, T, H, TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(on(C, S), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), truck(T).
taskTBA(do_clear(C, P), TT2) :- m10_do_load_truck(do_load_truck, C, S, P, T, H, TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(on(C, S), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT, surface(C).
taskTBA(lift_wrapper(H, C, S, P), TT3) :- m10_do_load_truck(do_load_truck, C, S, P, T, H, TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(on(C, S), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), causable(do_clear(C, P), TT2, TT3), TT3 >= TT2.
taskTBA(load_wrapper(H, C, T, P), TT4) :- m10_do_load_truck(do_load_truck, C, S, P, T, H, TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(on(C, S), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), causable(lift_wrapper(H, C, S, P), TT3, TT4), TT4 >= TT3, truck(T).

taskTBA(do_clear(S, P), TT) :- m11_do_unload_truck(do_unload_truck, C, S, P, T, H, TT), in_state(in(C, T), TT), in_state(at(T, P), TT), in_state(at(H, P), TT), in_state(at(S, P), TT), surface(S), hoist(H), truck(T), crate(C), place(P).
taskTBA(unload_wrapper(H, C, T, P), TT2) :- m11_do_unload_truck(do_unload_truck, C, S, P, T, H, TT), in_state(in(C, T), TT), in_state(at(T, P), TT), in_state(at(H, P), TT), in_state(at(S, P), TT), surface(S), hoist(H), truck(T), crate(C), place(P), causable(do_clear(S, P), TT, TT2), TT2 >= TT.
taskTBA(drop_wrapper(H, C, S, P), TT3) :- m11_do_unload_truck(do_unload_truck, C, S, P, T, H, TT), in_state(in(C, T), TT), in_state(at(T, P), TT), in_state(at(H, P), TT), in_state(at(S, P), TT), surface(S), hoist(H), truck(T), crate(C), place(P), causable(unload_wrapper(H, C, T, P), TT2, TT3), TT3 >= TT2.

0 { taskTBA(drive(X, Y, Z), TT) : m_drive_wrapper(drive_wrapper, X, Y, Z, TT), in_state(at(X, Y), TT), place(Y), truck(X), place(Z) } 1 :- time(TT).

0 { taskTBA(lift(X, Y, Z, P), TT) : m_lift_wrapper(lift_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(available(X), TT), in_state(at(Y, P), TT), in_state(on(Y, Z), TT), in_state(clear(Y), TT), surface(Z), place(P), hoist(X), crate(Y) } 1 :- time(TT).

0 { taskTBA(drop(X, Y, Z, P), TT) : m_drop_wrapper(drop_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(clear(Z), TT), in_state(lifting(X, Y), TT), surface(Z), place(P), hoist(X), crate(Y) } 1 :- time(TT).

0 { taskTBA(load(X, Y, Z, P), TT) : m_load_wrapper(load_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(lifting(X, Y), TT), truck(Z), place(P), hoist(X), crate(Y) } 1 :- time(TT).

0 { taskTBA(unload(X, Y, Z, P), TT) : m_unload_wrapper(unload_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(available(X), TT), in_state(in(Y, Z), TT), truck(Z), place(P), hoist(X), crate(Y) } 1 :- time(TT).

% Causable rules for compound tasks

causable(do_put_on(C, S2), TT, TT2) :- m0_do_put_on(do_put_on, C, S2, TT), in_state(on(C, S2), TT), surface(S2), crate(C), crate(C), surface(S2), causable(nop, TT, TT2), TT2 >= TT.
causable(do_put_on(C, S2), TT, TT5) :- m1_do_put_on(do_put_on, C, S2, P, H, TT), in_state(at(C, P), TT), crate(C), place(P), crate(C), surface(S2), causable(do_clear(C, P), TT, TT2), TT2 >= TT, causable(do_clear(S2, P), TT2, TT3), TT3 >= TT2, causable(do_lift_crate(C, P, H), TT3, TT4), TT4 >= TT3, causable(drop_wrapper(H, C, S2, P), TT4, TT5), TT5 >= TT4.
causable(do_put_on(C, S2), TT, TT5) :- m2_do_put_on(do_put_on, C, S2, P, T, H, TT), in_state(in(C, T), TT), truck(T), crate(C), crate(C), surface(S2), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT, causable(do_clear(S2, P), TT2, TT3), TT3 >= TT2, causable(unload_wrapper(H, C, T, P), TT3, TT4), TT4 >= TT3, causable(drop_wrapper(H, C, S2, P), TT4, TT5), TT5 >= TT4.
causable(do_put_on(C, S2), TT, TT4) :- m3_do_put_on(do_put_on, C, S2, S1, P1, T, P2, TT), crate(C), surface(S2), causable(do_load_truck(C, S1, P1, T), TT, TT2), TT2 >= TT, causable(drive_wrapper(T, P1, P2), TT2, TT3), TT3 >= TT2, causable(do_unload_truck(C, S2, P2, T), TT3, TT4), TT4 >= TT3.

causable(do_clear(S1, P1), TT, TT2) :- m4_do_clear(do_clear, S1, P1, TT), in_state(clear(S1), TT), in_state(at(S1, P1), TT), surface(S1), place(P1), surface(S1), place(P1), causable(nop, TT, TT2), TT2 >= TT.
causable(do_clear(S1, P1), TT, TT5) :- m5_do_clear(do_clear, S1, P1, C, T, H1, TT), not in_state(clear(S1), TT), in_state(on(C, S1), TT), in_state(at(S1, P1), TT), in_state(at(H1, P1), TT), hoist(H1), surface(S1), crate(C), place(P1), surface(S1), place(P1), causable(do_clear(C, P1), TT, TT2), TT2 >= TT, causable(lift_wrapper(H1, C, S1, P1), TT2, TT3), TT3 >= TT2, causable(do_get_truck(T, P1), TT3, TT4), TT4 >= TT3, causable(load_wrapper(H1, C, T, P1), TT4, TT5), TT5 >= TT4.

causable(do_get_truck(T, P1), TT, TT2) :- m6_do_get_truck(do_get_truck, T, P1, TT), in_state(at(T, P1), TT), truck(T), place(P1), truck(T), place(P1), causable(nop, TT, TT2), TT2 >= TT.
causable(do_get_truck(T, P1), TT, TT2) :- m7_do_get_truck(do_get_truck, T, P1, P2, TT), not in_state(at(T, P1), TT), truck(T), place(P1), truck(T), place(P1), causable(drive_wrapper(T, P2, P1), TT, TT2), TT2 >= TT.

causable(do_lift_crate(C, P, H), TT, TT3) :- m8_do_lift_crate(do_lift_crate, C, P, H, T, TT), in_state(in(C, T), TT), in_state(at(H, P), TT), truck(T), place(P), crate(C), hoist(H), crate(C), place(P), hoist(H), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT, causable(unload_wrapper(H, C, T, P), TT2, TT3), TT3 >= TT2.
causable(do_lift_crate(C, P, H), TT, TT2) :- m9_do_lift_crate(do_lift_crate, C, P, H, S, TT), in_state(on(C, S), TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), crate(C), place(P), hoist(H), causable(lift_wrapper(H, C, S, P), TT, TT2), TT2 >= TT.

causable(do_load_truck(C, S, P, T), TT, TT5) :- m10_do_load_truck(do_load_truck, C, S, P, T, H, TT), in_state(at(C, P), TT), in_state(at(S, P), TT), in_state(on(C, S), TT), in_state(at(H, P), TT), hoist(H), crate(C), place(P), surface(S), crate(C), surface(S), place(P), truck(T), causable(do_get_truck(T, P), TT, TT2), TT2 >= TT, causable(do_clear(C, P), TT2, TT3), TT3 >= TT2, causable(lift_wrapper(H, C, S, P), TT3, TT4), TT4 >= TT3, causable(load_wrapper(H, C, T, P), TT4, TT5), TT5 >= TT4.

causable(do_unload_truck(C, S, P, T), TT, TT4) :- m11_do_unload_truck(do_unload_truck, C, S, P, T, H, TT), in_state(in(C, T), TT), in_state(at(T, P), TT), in_state(at(H, P), TT), in_state(at(S, P), TT), surface(S), hoist(H), truck(T), crate(C), place(P), crate(C), surface(S), place(P), truck(T), causable(do_clear(S, P), TT, TT2), TT2 >= TT, causable(unload_wrapper(H, C, T, P), TT2, TT3), TT3 >= TT2, causable(drop_wrapper(H, C, S, P), TT3, TT4), TT4 >= TT3.

causable(drive_wrapper(X, Y, Z), TT, TT2) :- m_drive_wrapper(drive_wrapper, X, Y, Z, TT), in_state(at(X, Y), TT), place(Y), truck(X), truck(X), place(Y), place(Z), causable(drive(X, Y, Z), TT, TT2), TT2 >= TT.

causable(lift_wrapper(X, Y, Z, P), TT, TT2) :- m_lift_wrapper(lift_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(available(X), TT), in_state(at(Y, P), TT), in_state(on(Y, Z), TT), in_state(clear(Y), TT), surface(Z), place(P), hoist(X), crate(Y), hoist(X), crate(Y), surface(Z), place(P), causable(lift(X, Y, Z, P), TT, TT2), TT2 >= TT.

causable(drop_wrapper(X, Y, Z, P), TT, TT2) :- m_drop_wrapper(drop_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(clear(Z), TT), in_state(lifting(X, Y), TT), surface(Z), place(P), hoist(X), crate(Y), hoist(X), crate(Y), surface(Z), place(P), causable(drop(X, Y, Z, P), TT, TT2), TT2 >= TT.

causable(load_wrapper(X, Y, Z, P), TT, TT2) :- m_load_wrapper(load_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(lifting(X, Y), TT), truck(Z), place(P), hoist(X), crate(Y), hoist(X), crate(Y), truck(Z), place(P), causable(load(X, Y, Z, P), TT, TT2), TT2 >= TT.

causable(unload_wrapper(X, Y, Z, P), TT, TT2) :- m_unload_wrapper(unload_wrapper, X, Y, Z, P, TT), in_state(at(X, P), TT), in_state(at(Z, P), TT), in_state(available(X), TT), in_state(in(Y, Z), TT), truck(Z), place(P), hoist(X), crate(Y), hoist(X), crate(Y), truck(Z), place(P), causable(unload(X, Y, Z, P), TT, TT2), TT2 >= TT.
