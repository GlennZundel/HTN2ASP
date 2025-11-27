% Problem-specific facts

% Object declarations
factory(last_factory).
factory(factory_0).
location(location_0).
resource(resource_0).
factory(factory_1).
location(location_1).
resource(resource_1).
resource(resource_f_1_0).
location(last_location).

% Initial state
in_state(truck_at(location_0), 0).
in_state(factory_at(factory_0, location_0), 0).
in_state(factory_constructed(factory_0), 0).
in_state(factory_without_demands(factory_0), 0).
in_state(produces(factory_0, resource_0), 0).
in_state(demands(last_factory, resource_1), 0).
in_state(location_free(last_location), 0).
in_state(connected(location_1, location_0), 0).
in_state(connected(location_0, location_1), 0).
in_state(produces(factory_1, resource_1), 0).
in_state(location_free(location_1), 0).
in_state(demands(factory_1, resource_0), 0).
in_state(fuses(resource_f_1_0, resource_0, resource_0), 0).
in_state(connected(location_1, last_location), 0).
in_state(connected(last_location, location_1), 0).

% HTN Tasks
taskTBA(construct_factory(last_factory, last_location), 0).

% Goal verification
plan_found :- causable(construct_factory(last_factory, last_location), 0, T1), T1 >= 0, time(T1).