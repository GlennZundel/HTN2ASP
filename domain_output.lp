% Primitive tasks (actions)

causable(construct(F, R, L), T, T+1) :- taskTBA(construct(F, R, L), T), factory(F), resource(R), location(L).
out_state(resource_at(R, L), T+1) :- taskTBA(construct(F, R, L), T).
out_state(location_free(L), T+1) :- taskTBA(construct(F, R, L), T).
in_state(factory_at(F, L), T+1) :- taskTBA(construct(F, R, L), T).
in_state(factory_constructed(F), T+1) :- taskTBA(construct(F, R, L), T).

causable(fuse(R, R1, R2, L), T, T+1) :- taskTBA(fuse(R, R1, R2, L), T), resource(R), resource(R1), resource(R2), location(L).
out_state(resource_at(R1, L), T+1) :- taskTBA(fuse(R, R1, R2, L), T).
out_state(resource_at(R2, L), T+1) :- taskTBA(fuse(R, R1, R2, L), T).
in_state(resource_at(R, L), T+1) :- taskTBA(fuse(R, R1, R2, L), T).

causable(produce_without_demands(R, F, L), T, T+1) :- taskTBA(produce_without_demands(R, F, L), T), resource(R), factory(F), location(L).
in_state(resource_at(R, L), T+1) :- taskTBA(produce_without_demands(R, F, L), T).

causable(produce(R, RD, F, L), T, T+1) :- taskTBA(produce(R, RD, F, L), T), resource(R), resource(RD), factory(F), location(L).
out_state(resource_at(RD, L), T+1) :- taskTBA(produce(R, RD, F, L), T).
in_state(resource_at(R, L), T+1) :- taskTBA(produce(R, RD, F, L), T).

causable(pickup(R, L), T, T+1) :- taskTBA(pickup(R, L), T), resource(R), location(L).
out_state(resource_at(R, L), T+1) :- taskTBA(pickup(R, L), T).
in_state(resource_in_truck(R), T+1) :- taskTBA(pickup(R, L), T).

causable(drop(R, L), T, T+1) :- taskTBA(drop(R, L), T), resource(R), location(L).
out_state(resource_in_truck(R), T+1) :- taskTBA(drop(R, L), T).
in_state(resource_at(R, L), T+1) :- taskTBA(drop(R, L), T).

causable(move(L1, L2), T, T+1) :- taskTBA(move(L1, L2), T), location(L1), location(L2).
out_state(truck_at(L1), T+1) :- taskTBA(move(L1, L2), T).
in_state(truck_at(L2), T+1) :- taskTBA(move(L1, L2), T).

% Method selection rules

m_factory_already_constructed(F, L, T) :- taskTBA(construct_factory(F, L), T), factory(F), location(L).
m_construct_factory(F, R, L, T) :- taskTBA(construct_factory(F, L), T), factory(F), location(L), resource(R).

m_resource_there(R, L, T) :- taskTBA(get_resource(R, L), T), location(L), resource(R).
m_get_resources_and_fuse(R, R1, R2, L, T) :- taskTBA(get_resource(R, L), T), location(L), resource(R), resource(R1), resource(R2).
m_get_resource(R, F, FL, L, T) :- taskTBA(get_resource(R, L), T), factory(F), location(FL), location(L), resource(R).
m_produce_resource(R, F, L, T) :- taskTBA(produce_resource(R), T), factory(F), location(L), resource(R).
m_get_and_produce_resource(R, RD, F, L, T) :- taskTBA(produce_resource(R), T), factory(F), location(L), resource(R), resource(RD).

m_deliver_resource(R, LS, LE, T) :- taskTBA(deliver_resource(R, LE), T), location(LE), location(LS), resource(R).

m_goto(L1, L2, LE, T) :- taskTBA(goto(LE), T), location(L1), location(L2), location(LE).
m_already_there(L, T) :- taskTBA(goto(L), T), location(L).

m_construct_wrapper(F, R, L, T) :- taskTBA(construct_wrapper(F, R, L), T), factory(F), location(L), resource(R).

m_fuse_wrapper(R, R1, R2, L, T) :- taskTBA(fuse_wrapper(R, R1, R2, L), T), location(L), resource(R), resource(R1), resource(R2).

m_produce_without_demands_wrapper(R, F, L, T) :- taskTBA(produce_without_demands_wrapper(R, F, L), T), factory(F), location(L), resource(R).

m_produce_wrapper(R, RD, F, L, T) :- taskTBA(produce_wrapper(R, RD, F, L), T), factory(F), location(L), resource(R), resource(RD).

m_pickup_wrapper(R, L, T) :- taskTBA(pickup_wrapper(R, L), T), location(L), resource(R).

m_drop_wrapper(R, L, T) :- taskTBA(drop_wrapper(R, L), T), location(L), resource(R).

m_move_wrapper(L1, L2, T) :- taskTBA(move_wrapper(L1, L2), T), location(L1), location(L2).

% Subtask rules


taskTBA(get_resource(R, L), T) :- m_construct_factory(F, R, L, T), in_state(demands(F, R), T), in_state(location_free(L), T), not in_state(factory_constructed(F), T), resource(R), factory(F), location(L).
taskTBA(construct_wrapper(F, R, L), T2) :- m_construct_factory(F, R, L, T), in_state(demands(F, R), T), in_state(location_free(L), T), not in_state(factory_constructed(F), T), resource(R), factory(F), location(L), causable(get_resource(R, L), T, T2), T2 >= T.


taskTBA(get_resource(R1, L), T) :- m_get_resources_and_fuse(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), resource(R), resource(R2), resource(R1), location(L).
taskTBA(get_resource(R2, L), T2) :- m_get_resources_and_fuse(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), resource(R), resource(R2), resource(R1), causable(get_resource(R1, L), T, T2), T2 >= T, location(L).
taskTBA(fuse_wrapper(R, R1, R2, L), T3) :- m_get_resources_and_fuse(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), resource(R), resource(R2), resource(R1), causable(get_resource(R2, L), T2, T3), T3 >= T2, location(L).

taskTBA(construct_factory(F, FL), T) :- m_get_resource(R, F, FL, L, T), in_state(produces(F, R), T), resource(R), factory(F), location(FL).
taskTBA(produce_resource(R), T2) :- m_get_resource(R, F, FL, L, T), in_state(produces(F, R), T), resource(R), factory(F), causable(construct_factory(F, FL), T, T2), T2 >= T.
taskTBA(deliver_resource(R, L), T3) :- m_get_resource(R, F, FL, L, T), in_state(produces(F, R), T), resource(R), factory(F), causable(produce_resource(R), T2, T3), T3 >= T2, location(L).

taskTBA(produce_without_demands_wrapper(R, F, L), T) :- m_produce_resource(R, F, L, T), in_state(produces(F, R), T), in_state(factory_at(F, L), T), in_state(factory_without_demands(F), T), resource(R), factory(F), location(L).

taskTBA(get_resource(RD, L), T) :- m_get_and_produce_resource(R, RD, F, L, T), in_state(produces(F, R), T), in_state(demands(F, RD), T), in_state(factory_at(F, L), T), resource(R), factory(F), resource(RD), location(L).
taskTBA(produce_wrapper(R, RD, F, L), T2) :- m_get_and_produce_resource(R, RD, F, L, T), in_state(produces(F, R), T), in_state(demands(F, RD), T), in_state(factory_at(F, L), T), resource(R), factory(F), resource(RD), location(L), causable(get_resource(RD, L), T, T2), T2 >= T.

taskTBA(goto(LS), T) :- m_deliver_resource(R, LS, LE, T), in_state(resource_at(R, LS), T), resource(R), location(LS).
taskTBA(pickup_wrapper(R, LS), T2) :- m_deliver_resource(R, LS, LE, T), in_state(resource_at(R, LS), T), resource(R), location(LS), causable(goto(LS), T, T2), T2 >= T.
taskTBA(goto(LE), T3) :- m_deliver_resource(R, LS, LE, T), in_state(resource_at(R, LS), T), resource(R), location(LS), causable(pickup_wrapper(R, LS), T2, T3), T3 >= T2, location(LE).
taskTBA(drop_wrapper(R, LE), T4) :- m_deliver_resource(R, LS, LE, T), in_state(resource_at(R, LS), T), resource(R), location(LS), causable(goto(LE), T3, T4), T4 >= T3, location(LE).

taskTBA(move_wrapper(L1, L2), T) :- m_goto(L1, L2, LE, T), in_state(truck_at(L1), T), in_state(connected(L1, L2), T), location(L1), location(L2).
taskTBA(goto(LE), T2) :- m_goto(L1, L2, LE, T), in_state(truck_at(L1), T), in_state(connected(L1, L2), T), location(L1), location(L2), causable(move_wrapper(L1, L2), T, T2), T2 >= T, location(LE).


0 { taskTBA(construct(F, R, L), T) : m_construct_wrapper(F, R, L, T), in_state(location_free(L), T), in_state(demands(F, R), T), in_state(resource_at(R, L), T), resource(R), factory(F), location(L) } 1 :- time(T).

0 { taskTBA(fuse(R, R1, R2, L), T) : m_fuse_wrapper(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), in_state(resource_at(R1, L), T), in_state(resource_at(R2, L), T), resource(R), resource(R2), resource(R1), location(L) } 1 :- time(T).

0 { taskTBA(produce_without_demands(R, F, L), T) : m_produce_without_demands_wrapper(R, F, L, T), in_state(produces(F, R), T), in_state(factory_without_demands(F), T), in_state(factory_at(F, L), T), resource(R), factory(F), location(L) } 1 :- time(T).

0 { taskTBA(produce(R, RD, F, L), T) : m_produce_wrapper(R, RD, F, L, T), in_state(produces(F, R), T), in_state(demands(F, RD), T), in_state(factory_at(F, L), T), in_state(resource_at(RD, L), T), resource(R), factory(F), resource(RD), location(L) } 1 :- time(T).

0 { taskTBA(pickup(R, L), T) : m_pickup_wrapper(R, L, T), in_state(resource_at(R, L), T), in_state(truck_at(L), T), resource(R), location(L) } 1 :- time(T).

0 { taskTBA(drop(R, L), T) : m_drop_wrapper(R, L, T), in_state(truck_at(L), T), in_state(resource_in_truck(R), T), resource(R), location(L) } 1 :- time(T).

0 { taskTBA(move(L1, L2), T) : m_move_wrapper(L1, L2, T), in_state(truck_at(L1), T), in_state(connected(L1, L2), T), location(L1), location(L2) } 1 :- time(T).

% Causable rules for compound tasks

causable(construct_factory(F, L), T, T) :- m_factory_already_constructed(F, L, T), in_state(factory_at(F, L), T), factory(F), location(L), factory(F), location(L).
causable(construct_factory(F, L), T, T3) :- m_construct_factory(F, R, L, T), in_state(demands(F, R), T), in_state(location_free(L), T), not in_state(factory_constructed(F), T), resource(R), factory(F), location(L), factory(F), location(L), causable(get_resource(R, L), T, T2), T2 >= T, causable(construct_wrapper(F, R, L), T2, T3), T3 >= T2.

causable(get_resource(R, L), T, T) :- m_resource_there(R, L, T), in_state(resource_at(R, L), T), resource(R), location(L), resource(R), location(L).
causable(get_resource(R, L), T, T4) :- m_get_resources_and_fuse(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), resource(R), resource(R2), resource(R1), resource(R), location(L), causable(get_resource(R1, L), T, T2), T2 >= T, causable(get_resource(R2, L), T2, T3), T3 >= T2, causable(fuse_wrapper(R, R1, R2, L), T3, T4), T4 >= T3.
causable(get_resource(R, L), T, T4) :- m_get_resource(R, F, FL, L, T), in_state(produces(F, R), T), resource(R), factory(F), resource(R), location(L), causable(construct_factory(F, FL), T, T2), T2 >= T, causable(produce_resource(R), T2, T3), T3 >= T2, causable(deliver_resource(R, L), T3, T4), T4 >= T3.

causable(produce_resource(R), T, T2) :- m_produce_resource(R, F, L, T), in_state(produces(F, R), T), in_state(factory_at(F, L), T), in_state(factory_without_demands(F), T), resource(R), factory(F), location(L), resource(R), causable(produce_without_demands_wrapper(R, F, L), T, T2), T2 >= T.
causable(produce_resource(R), T, T3) :- m_get_and_produce_resource(R, RD, F, L, T), in_state(produces(F, R), T), in_state(demands(F, RD), T), in_state(factory_at(F, L), T), resource(R), factory(F), resource(RD), location(L), resource(R), causable(get_resource(RD, L), T, T2), T2 >= T, causable(produce_wrapper(R, RD, F, L), T2, T3), T3 >= T2.

causable(deliver_resource(R, LE), T, T5) :- m_deliver_resource(R, LS, LE, T), in_state(resource_at(R, LS), T), resource(R), location(LS), resource(R), location(LE), causable(goto(LS), T, T2), T2 >= T, causable(pickup_wrapper(R, LS), T2, T3), T3 >= T2, causable(goto(LE), T3, T4), T4 >= T3, causable(drop_wrapper(R, LE), T4, T5), T5 >= T4.

causable(goto(LE), T, T3) :- m_goto(L1, L2, LE, T), in_state(truck_at(L1), T), in_state(connected(L1, L2), T), location(L1), location(L2), location(LE), causable(move_wrapper(L1, L2), T, T2), T2 >= T, causable(goto(LE), T2, T3), T3 >= T2.
causable(goto(L), T, T) :- m_already_there(L, T), in_state(truck_at(L), T), location(L), location(L).

causable(construct_wrapper(F, R, L), T, T2) :- m_construct_wrapper(F, R, L, T), in_state(location_free(L), T), in_state(demands(F, R), T), in_state(resource_at(R, L), T), resource(R), factory(F), location(L), factory(F), resource(R), location(L), causable(construct(F, R, L), T, T2), T2 >= T.

causable(fuse_wrapper(R, R1, R2, L), T, T2) :- m_fuse_wrapper(R, R1, R2, L, T), in_state(fuses(R, R1, R2), T), in_state(resource_at(R1, L), T), in_state(resource_at(R2, L), T), resource(R), resource(R2), resource(R1), location(L), resource(R), resource(R1), resource(R2), location(L), causable(fuse(R, R1, R2, L), T, T2), T2 >= T.

causable(produce_without_demands_wrapper(R, F, L), T, T2) :- m_produce_without_demands_wrapper(R, F, L, T), in_state(produces(F, R), T), in_state(factory_without_demands(F), T), in_state(factory_at(F, L), T), resource(R), factory(F), location(L), resource(R), factory(F), location(L), causable(produce_without_demands(R, F, L), T, T2), T2 >= T.

causable(produce_wrapper(R, RD, F, L), T, T2) :- m_produce_wrapper(R, RD, F, L, T), in_state(produces(F, R), T), in_state(demands(F, RD), T), in_state(factory_at(F, L), T), in_state(resource_at(RD, L), T), resource(R), factory(F), resource(RD), location(L), resource(R), resource(RD), factory(F), location(L), causable(produce(R, RD, F, L), T, T2), T2 >= T.

causable(pickup_wrapper(R, L), T, T2) :- m_pickup_wrapper(R, L, T), in_state(resource_at(R, L), T), in_state(truck_at(L), T), resource(R), location(L), resource(R), location(L), causable(pickup(R, L), T, T2), T2 >= T.

causable(drop_wrapper(R, L), T, T2) :- m_drop_wrapper(R, L, T), in_state(truck_at(L), T), in_state(resource_in_truck(R), T), resource(R), location(L), resource(R), location(L), causable(drop(R, L), T, T2), T2 >= T.

causable(move_wrapper(L1, L2), T, T2) :- m_move_wrapper(L1, L2, T), in_state(truck_at(L1), T), in_state(connected(L1, L2), T), location(L1), location(L2), location(L1), location(L2), causable(move(L1, L2), T, T2), T2 >= T.
