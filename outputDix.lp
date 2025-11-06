% ================================================================
% == ASP-Übersetzung für die Roboter-Domäne                    ==
% == Basierend auf der Methodik von Dix, Kuter & Nau (2005)   ==
% ================================================================

% ################################################################
% ## 1. KONSTANTEN UND ATOM-DEKLARATIONEN (aus Domain & Problem)##
% ################################################################

package(o1).
room(c; r1; r2; r3).
roomdoor(d12; d13; d01).

% --- Deklaration aller möglichen Atome für das Grounding ---

atom(armempty).
atom(closed(D)) :- roomdoor(D).
atom(door(LOC1, LOC2, D)) :- room(LOC1), room(LOC2), roomdoor(D).
atom(goal_in(OBJ, LOC)) :- package(OBJ), room(LOC).
atom(holding(OBJ)) :- package(OBJ).
atom(in(OBJ, LOC)) :- package(OBJ), room(LOC).
atom(rloc(LOC)) :- room(LOC).


% ################################################################
% ## 2. INITIALZUSTAND - Trans(S)                               ##
% ################################################################


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


% ################################################################
% ## 3. ZIELAUFGABE(N) - Trans(t)                               ##
% ################################################################


% Die initiale Aufgabe zur Zeit 0 ist 'achieve-goals'.
taskTBA(achieve_goals, 0).


% ################################################################
% ## 4. OPERATOREN (Aktionen) - Trans(OP)                       ##
% ################################################################


% --- Operator 'move' ---
in_state(rloc(LOC2), T+1) :- taskTBA(move(LOC1, LOC2, D), T).
out_state(rloc(LOC1), T+1)   :- taskTBA(move(LOC1, LOC2, D), T).

% --- Operator 'open' ---
out_state(closed(D), T+1)   :- taskTBA(open(LOC1, LOC2, D), T).

% --- Operator 'pickup' ---
out_state(in(OBJ, LOC), T+1)   :- taskTBA(pickup(OBJ, LOC), T).
out_state(armempty, T+1)   :- taskTBA(pickup(OBJ, LOC), T).
in_state(holding(OBJ), T+1) :- taskTBA(pickup(OBJ, LOC), T).

% --- Operator 'putdown' ---
out_state(holding(OBJ), T+1)   :- taskTBA(putdown(OBJ, LOC), T).
in_state(armempty, T+1) :- taskTBA(putdown(OBJ, LOC), T).
in_state(in(OBJ, LOC), T+1) :- taskTBA(putdown(OBJ, LOC), T).


% ################################################################
% ## 5. METHODEN - Trans(METH)                                  ##
% ################################################################


% --- 5.1 Übersetzung von "abstrakten" zu "primitiven" Aufgaben ---


% Methode für 'pickup_abstract' -> 'pickup'
method_pickup_abstract(OBJ, T) :- taskTBA(pickup_abstract(OBJ), T).
taskTBA(pickup(OBJ, LOC), T) :-
    method_pickup_abstract(OBJ, T),
    in_state(armempty, T),
    in_state(rloc(LOC), T),
    in_state(in(OBJ, LOC), T). % Preconditions der 'pickup'-Aktion
causable(pickup_abstract(OBJ), T_start, T_end) :-
    method_pickup_abstract(OBJ, T_start),
    package(OBJ),
    room(LOC),
    in_state(armempty, T_start),
    in_state(rloc(LOC), T_start),
    in_state(in(OBJ, LOC), T_start),
    causable(pickup(OBJ, LOC), T_start, T_end).

% Methode für 'putdown_abstract' -> 'putdown'
method_putdown_abstract(T) :- taskTBA(putdown_abstract, T).
taskTBA(putdown(OBJ, LOC), T) :-
    method_putdown_abstract(T),
    in_state(rloc(LOC), T),
    in_state(holding(OBJ), T),
    in_state(goal_in(OBJ, LOC), T). % Preconditions der 'putdown'-Aktion
causable(putdown_abstract, T_start, T_end) :-
    method_putdown_abstract(T_start),
    package(OBJ),
    room(LOC),
    in_state(rloc(LOC), T_start),
    in_state(holding(OBJ), T_start),
    in_state(goal_in(OBJ, LOC), T_start),
    causable(putdown(OBJ, LOC), T_start, T_end).

% Methode für 'move_abstract' -> 'move'
method_move_abstract(T) :- taskTBA(move_abstract, T).
taskTBA(move(LOC1, LOC2, D), T) :-
    method_move_abstract(T),
    in_state(rloc(LOC1), T),
    in_state(door(LOC1, LOC2, D), T),
    not in_state(closed(D), T). % Preconditions der 'move'-Aktion
causable(move_abstract, T_start, T_end) :-
    method_move_abstract(T_start),
    room(LOC1),
    room(LOC2),
    roomdoor(D),
    in_state(rloc(LOC1), T_start),
    in_state(door(LOC1, LOC2, D), T_start),
    not in_state(closed(D), T_start),
    causable(move(LOC1, LOC2, D), T_start, T_end).


% Methode für 'open_abstract' -> 'open'
method_open_abstract(T) :- taskTBA(open_abstract, T).
taskTBA(open(LOC1, LOC2, D), T) :-
    method_open_abstract(T),
    in_state(rloc(LOC1), T),
    in_state(door(LOC1, LOC2, D), T),
    in_state(closed(D), T). % Preconditions der 'open'-Aktion
causable(open_abstract, T_start, T_end) :-
    method_open_abstract(T_start),
    room(LOC1),
    room(LOC2),
    roomdoor(D),
    in_state(rloc(LOC1), T_start),
    in_state(door(LOC1, LOC2, D), T_start),
    in_state(closed(D), T_start),
    causable(open(LOC1, LOC2, D), T_start, T_end).


% --- 5.2 Übersetzung der Kontrollfluss-Methoden ---

% --- Methoden für 'achieve-goals' ---
% Nicht-deterministische Auswahl einer der möglichen Methoden
method_achieve_goals_pickup(T) :- taskTBA(achieve_goals, T), not method_achieve_goals_move(T), not method_achieve_goals_open(T), not method_finished(T).
method_achieve_goals_move(T) :- taskTBA(achieve_goals, T), not method_achieve_goals_pickup(T), not method_achieve_goals_open(T), not method_finished(T).
method_achieve_goals_open(T) :- taskTBA(achieve_goals, T), not method_achieve_goals_pickup(T), not method_achieve_goals_move(T), not method_finished(T).
method_finished(T) :- taskTBA(achieve_goals, T), not method_achieve_goals_pickup(T), not method_achieve_goals_move(T), not method_achieve_goals_open(T).

% Methode 'achieve-goals-pickup'
taskTBA(pickup_abstract(OBJ), T) :-
    method_achieve_goals_pickup(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T).
taskTBA(release, T2) :-
    method_achieve_goals_pickup(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T),
    causable(pickup_abstract(OBJ), T, T2), T2 >= T.
causable(achieve_goals, T, T3) :-
    method_achieve_goals_pickup(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(in(OBJ, LOC), T), not in_state(goal_in(OBJ, LOC), T),
    causable(release, T2, T3), T3 >= T2, causable(pickup_abstract(OBJ), T, T2).

% Methode 'achieve-goals-move'
taskTBA(move_abstract, T) :-
    method_achieve_goals_move(T).
taskTBA(achieve_goals, T2) :-
    method_achieve_goals_move(T),
    causable(move_abstract, T, T2), T2 >= T.
causable(achieve_goals, T, T3) :-
    method_achieve_goals_move(T),
    causable(achieve_goals, T2, T3), T3 >= T2, causable(move_abstract, T, T2).

% Methode 'achieve-goals-open'
taskTBA(open_abstract, T) :-
    method_achieve_goals_open(T).
taskTBA(achieve_goals, T2) :-
    method_achieve_goals_open(T),
    causable(open_abstract, T, T2), T2 >= T.
causable(achieve_goals, T, T3) :-
    method_achieve_goals_open(T),
    causable(achieve_goals, T2, T3), T3 >= T2, causable(open_abstract, T, T2).

% Methode 'finished'
causable(achieve_goals, T, T) :- method_finished(T).

% --- Methoden für 'release' ---
% Nicht-deterministische Auswahl einer der möglichen Methoden
method_release_putdown_abstract(T) :- taskTBA(release, T), not method_release_move(T), not method_release_open(T).
method_release_move(T) :- taskTBA(release, T), not method_release_putdown_abstract(T), not method_release_open(T).
method_release_open(T) :- taskTBA(release, T), not method_release_putdown_abstract(T), not method_release_move(T).

% Methode 'release-putdown_abstract'
taskTBA(putdown_abstract, T) :-
    method_release_putdown_abstract(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T).
taskTBA(achieve_goals, T2) :-
    method_release_putdown_abstract(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T),
    causable(putdown_abstract, T, T2), T2 >= T.
causable(release, T, T3) :-
    method_release_putdown_abstract(T), room(LOC), package(OBJ), in_state(rloc(LOC), T), in_state(holding(OBJ), T), in_state(goal_in(OBJ, LOC), T),
    causable(achieve_goals, T2, T3), T3 >= T2, causable(putdown_abstract, T, T2).

% Methode 'release-move'
taskTBA(move_abstract, T) :-
    method_release_move(T).
taskTBA(release, T2) :-
    method_release_move(T),
    causable(move_abstract, T, T2), T2 >= T.
causable(release, T, T3) :-
    method_release_move(T),
    causable(release, T2, T3), T3 >= T2, causable(move_abstract, T, T2).

% Methode 'release-open'
taskTBA(open_abstract, T) :-
    method_release_open(T).
taskTBA(release, T2) :-
    method_release_open(T),
    causable(open_abstract, T, T2), T2 >= T.
causable(release, T, T3) :-
    method_release_open(T),
    causable(release, T2, T3), T3 >= T2, causable(open_abstract, T, T2).


% ################################################################
% ## 6. CAUSABLE DEFINITION FÜR PRIMITIVE AUFGABEN              ##
% ################################################################

causable(move(LOC1, LOC2, D), T, T+1)   :- taskTBA(move(LOC1, LOC2, D), T).
causable(open(LOC1, LOC2, D), T, T+1)   :- taskTBA(open(LOC1, LOC2, D), T).
causable(pickup(OBJ, LOC), T, T+1)   :- taskTBA(pickup(OBJ, LOC), T).
causable(putdown(OBJ, LOC), T, T+1)   :- taskTBA(putdown(OBJ, LOC), T).


% ################################################################
% ## 7. DEKLARATION PRIMITIVER TASKS                            ##
% ################################################################

primitive_task(move(LOC1, LOC2, D)) :- room(LOC1), room(LOC2), roomdoor(D).
primitive_task(open(LOC1, LOC2, D)) :- room(LOC1), room(LOC2), roomdoor(D).
primitive_task(pickup(OBJ, LOC)) :- package(OBJ), room(LOC).
primitive_task(putdown(OBJ, LOC)) :- package(OBJ), room(LOC).


% ################################################################
% ## 8. ZIEL-ÜBERSETZUNG - Trans(G)                             ##
% ################################################################

% Die Ziele aus der :goal Sektion des Problems müssen erfüllt sein

% Für jedes Ziel: Es muss zu IRGENDEINEM Zeitpunkt erfüllt sein
goal_0_satisfied :- in_state(in(o1, r2), T), time(T).
goal_satisfied :- goal_0_satisfied.


% ################################################################
% ## 9. ERFOLGREICHE TERMINIERUNG - Trans(⊥)                    ##
% ################################################################

% Ein Plan ist erfolgreich, wenn:
% 1. Die initiale Aufgabe causable ist
% 2. Alle Ziele im Endzustand erfüllt sind

plan_found :- causable(achieve_goals, 0, T_final), goal_satisfied, time(T_final).

% Integrity Constraint: Es muss einen Plan geben
:- not plan_found.


% ################################################################
% ## 10. CONSTRAINTS                                             ##
% ################################################################

% Constraint 1: Nur eine primitive Aktion pro Zeitschritt
% (Verhindert gleichzeitige Ausführung mehrerer Aktionen)
