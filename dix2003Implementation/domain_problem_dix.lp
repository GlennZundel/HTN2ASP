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
atom(rloc(R)) :- room(R).
atom(in(O, R)) :- package(O), room(R).
atom(holding(O)) :- package(O).
atom(closed(D)) :- roomdoor(D).
atom(door(R1, R2, D)) :- room(R1), room(R2), roomdoor(D).
atom(goal_in(O, R)) :- package(O), room(R).


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


% --- Operator 'pickup' ---
out_state(in(O, L), T+1)   :- taskTBA(pickup(O, L), T).
out_state(armempty, T+1)    :- taskTBA(pickup(O, L), T).
in_state(holding(O), T+1) :- taskTBA(pickup(O, L), T).

% --- Operator 'putdown' ---
out_state(holding(O), T+1) :- taskTBA(putdown(O, L), T).
in_state(armempty, T+1)    :- taskTBA(putdown(O, L), T).
in_state(in(O, L), T+1)    :- taskTBA(putdown(O, L), T).

% --- Operator 'move' ---
out_state(rloc(L1), T+1) :- taskTBA(move(L1, L2, D), T).
in_state(rloc(L2), T+1)  :- taskTBA(move(L1, L2, D), T).

% --- Operator 'open' ---
out_state(closed(D), T+1) :- taskTBA(open(L1, L2, D), T).


% ################################################################
% ## 5. METHODEN - Trans(METH)                                  ##
% ################################################################


% --- 5.1 Übersetzung von "abstrakten" zu "primitiven" Aufgaben ---


% Methode für 'pickup_abstract' -> 'pickup'
method_pickup(O, T) :- taskTBA(pickup_abstract(O), T).
taskTBA(pickup(O, L), T) :-
    method_pickup(O, T),
    in_state(rloc(L), T),
    in_state(in(O, L), T),
    in_state(armempty, T). % Preconditions der 'pickup'-Aktion
causable(pickup_abstract(O), T_start, T_end) :-
    method_pickup(O, T_start), room(L),
    in_state(rloc(L), T_start), in_state(in(O, L), T_start), in_state(armempty, T_start),
    causable(pickup(O, L), T_start, T_end).

% Methode für 'putdown_abstract' -> 'putdown'
method_putdown(T) :- taskTBA(putdown_abstract, T).
taskTBA(putdown(O, L), T) :-
    method_putdown(T),
    in_state(rloc(L), T),
    in_state(holding(O), T),
    in_state(goal_in(O, L), T). % Preconditions der 'putdown'-Aktion
causable(putdown_abstract, T_start, T_end) :-
    method_putdown(T_start), package(O), room(L),
    in_state(rloc(L), T_start), in_state(holding(O), T_start), in_state(goal_in(O, L), T_start),
    causable(putdown(O, L), T_start, T_end).

% Methode für 'move_abstract' -> 'move'
method_move(T) :- taskTBA(move_abstract, T).
taskTBA(move(L1, L2, D), T) :-
    method_move(T),
    in_state(rloc(L1), T),
    in_state(door(L1, L2, D), T),
    not in_state(closed(D), T). % Preconditions der 'move'-Aktion
causable(move_abstract, T_start, T_end) :-
    method_move(T_start), room(L1;L2), roomdoor(D),
    in_state(rloc(L1), T_start), in_state(door(L1, L2, D), T_start), not in_state(closed(D), T_start),
    causable(move(L1, L2, D), T_start, T_end).

% Methode für 'open_abstract' -> 'open'
method_open(T) :- taskTBA(open_abstract, T).
taskTBA(open(L1, L2, D), T) :-
    method_open(T),
    in_state(rloc(L1), T),
    in_state(door(L1, L2, D), T),
    in_state(closed(D), T). % Preconditions der 'open'-Aktion
causable(open_abstract, T_start, T_end) :-
    method_open(T_start), room(L1;L2), roomdoor(D),
    in_state(rloc(L1), T_start), in_state(door(L1, L2, D), T_start), in_state(closed(D), T_start),
    causable(open(L1, L2, D), T_start, T_end).


% --- 5.2 Übersetzung der Kontrollfluss-Methoden ---

% --- Methoden für 'achieve-goals' ---
% Nicht-deterministische Auswahl einer der vier möglichen Methoden
method_ag_pickup(T)   :- taskTBA(achieve_goals, T), not method_ag_move(T), not method_ag_open(T), not method_ag_finished(T).
method_ag_move(T)     :- taskTBA(achieve_goals, T), not method_ag_pickup(T), not method_ag_open(T), not method_ag_finished(T).
method_ag_open(T)     :- taskTBA(achieve_goals, T), not method_ag_pickup(T), not method_ag_move(T), not method_ag_finished(T).
method_ag_finished(T) :- taskTBA(achieve_goals, T), not method_ag_pickup(T), not method_ag_move(T), not method_ag_open(T).


taskTBA(pickup_abstract(O), T) :-
    method_ag_pickup(T), room(L), package(O),
    in_state(rloc(L), T), in_state(in(O, L), T), not in_state(goal_in(O, L), T).
taskTBA(release, T2) :-
    method_ag_pickup(T), room(L), package(O),
    in_state(rloc(L), T), in_state(in(O, L), T), not in_state(goal_in(O, L), T),
    causable(pickup_abstract(O), T, T2), T2 >= T.
causable(achieve_goals, T, T3) :-
    method_ag_pickup(T), room(L), package(O),
    in_state(rloc(L), T), in_state(in(O, L), T), not in_state(goal_in(O, L), T),
    causable(release, T2, T3), T3 >= T2.

% Methode 'achieve-goals-move'
taskTBA(move_abstract, T)   :- method_ag_move(T).
taskTBA(achieve_goals, T2)  :- method_ag_move(T), causable(move_abstract, T, T2), T2 >= T.
causable(achieve_goals, T, T3) :- method_ag_move(T), causable(achieve_goals, T2, T3), T3 >= T2, causable(move_abstract, T, T2).

% Methode 'achieve-goals-open'
taskTBA(open_abstract, T)   :- method_ag_open(T).
taskTBA(achieve_goals, T2)  :- method_ag_open(T), causable(open_abstract, T, T2), T2 >= T.
causable(achieve_goals, T, T3) :- method_ag_open(T), causable(achieve_goals, T2, T3), T3 >= T2, causable(open_abstract, T, T2).

% Methode 'finished' (Basis-Fall der Rekursion)
% Wenn diese Methode gewählt wird, ist 'achieve_goals' sofort erfüllt, da es keine Subtasks gibt.
causable(achieve_goals, T, T) :- method_ag_finished(T).


% --- Methoden für 'release' ---
% Nicht-deterministische Auswahl einer der drei möglichen Methoden
method_rel_putdown(T) :- taskTBA(release, T), not method_rel_move(T), not method_rel_open(T).
method_rel_move(T)    :- taskTBA(release, T), not method_rel_putdown(T), not method_rel_open(T).
method_rel_open(T)    :- taskTBA(release, T), not method_rel_putdown(T), not method_rel_move(T).

% Methode 'release-putdown_abstract'
taskTBA(putdown_abstract, T) :-
    method_rel_putdown(T), room(L), package(O),
    in_state(rloc(L), T), in_state(holding(O), T), in_state(goal_in(O, L), T).
taskTBA(achieve_goals, T2) :-
    method_rel_putdown(T), room(L), package(O),
    in_state(rloc(L), T), in_state(holding(O), T), in_state(goal_in(O, L), T),
    causable(putdown_abstract, T, T2), T2 >= T.
causable(release, T, T3) :-
    method_rel_putdown(T), room(L), package(O),
    in_state(rloc(L), T), in_state(holding(O), T), in_state(goal_in(O, L), T),
    causable(achieve_goals, T2, T3), T3 >= T2, causable(putdown_abstract, T, T2).

% Methode 'release-move'
taskTBA(move_abstract, T) :- method_rel_move(T).
taskTBA(release, T2)      :- method_rel_move(T), causable(move_abstract, T, T2), T2 >= T.
causable(release, T, T3)  :- method_rel_move(T), causable(release, T2, T3), T3 >= T2, causable(move_abstract, T, T2).

% Methode 'release-open'
taskTBA(open_abstract, T) :- method_rel_open(T).
taskTBA(release, T2)      :- method_rel_open(T), causable(open_abstract, T, T2), T2 >= T.
causable(release, T, T3)  :- method_rel_open(T), causable(release, T2, T3), T3 >= T2, causable(open_abstract, T, T2).


% ################################################################
% ## 6. CAUSABLE DEFINITION FÜR PRIMITIVE AUFGABEN              ##
% ################################################################

% Eine primitive Aufgabe ist 'causable' einen Zeitschritt nachdem sie gestartet wurde.
% Dies inkrementiert die Zeit.
causable(pickup(O, L), T, T+1)   :- taskTBA(pickup(O, L), T).
causable(putdown(O, L), T, T+1)  :- taskTBA(putdown(O, L), T).
causable(move(L1, L2, D), T, T+1) :- taskTBA(move(L1, L2, D), T).
causable(open(L1, L2, D), T, T+1) :- taskTBA(open(L1, L2, D), T).