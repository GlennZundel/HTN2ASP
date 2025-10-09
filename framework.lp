% Zu jedem Zeitpunkt T (außer dem letzten) findet genau eine Aktion statt.
% Diese Regel generiert mögliche Pläne.
1 { occurs(A, T) : action(A) } 1 :- time(T), T < n.

% --- a) Klassische Zustandsübergänge und Vorbedingungen ---

% Inertia-Regel: Ein Fluent bleibt wahr, wenn es nicht durch eine Aktion gelöscht wird.
holds(F, T+1) :-
    holds(F, T),
    occurs(A, T),
    not deletes(A, F),
    time(T), T < n.

% Effekt-Regel: Ein Fluent wird wahr, wenn eine Aktion es hinzufügt.
holds(F, T+1) :-
    occurs(A, T),
    adds(A, F),
    time(T), T < n.

% Declare static predicates (predicates that never change)
static(door(_,_,_)).
static(goal_in(_,_)).

% Test (Constraint): Eine Aktion darf nur stattfinden, wenn ihre Vorbedingungen erfüllt sind.
% For dynamic predicates (fluents)
:- occurs(A, T), precondition(A, F), not static(F), not holds(F, T).
% For static predicates
:- occurs(A, T), precondition(A, F), static(F), not F.

:- occurs(A, T), neg_precondition(A, F), not static(F), holds(F, T).
:- occurs(A, T), neg_precondition(A, F), static(F), F.

% --- b) HTN-Zerlegung und Zielerreichung ---

% Base Case: Eine primitive Aufgabe T ist "erledigt" (achieved),
% wenn die entsprechende Aktion zur Zeit T_start stattfindet.
% Der Vorgang dauert einen Zeitschritt.
achieves(T, T_start, T_start + 1) :-
    occurs(T, T_start),
    primitive(T).

% Recursive Case: Eine abstrakte Aufgabe T ist "erledigt" (achieved) von T_start bis T_end,
% wenn es eine Methode M gibt, die T zerlegt, deren Vorbedingungen zur Zeit T_start
% erfüllt sind und deren Kette von Teilaufgaben das Zeitintervall [T_start, T_end] füllt.
achieves(T, T_start, T_end) :-
    abstract(T),
    decomposes(M, T),
    % Prüfe Vorbedingungen der Methode zur Startzeit
    % Erledige die Kette der Teilaufgaben
    achieves_subtasks(M, 1, T_start, T_end).

%%Tests
% Method preconditions for dynamic predicates
:- achieves(T, T_start, _),
   decomposes(M, T),
   achieves_subtasks(M, 1, T_start, _),
   method_precondition(M, F),
   not static(F),
   not holds(F, T_start).

% Method preconditions for static predicates
:- achieves(T, T_start, _),
   decomposes(M, T),
   achieves_subtasks(M, 1, T_start, _),
   method_precondition(M, F),
   static(F),
   not F.

% Negative method preconditions for dynamic predicates
:- achieves(T, T_start, _),
   decomposes(M, T),
   achieves_subtasks(M, 1, T_start, _),
   neg_method_precondition(M, F),
   not static(F),
   holds(F, T_start).

% Negative method preconditions for static predicates
:- achieves(T, T_start, _),
   decomposes(M, T),
   achieves_subtasks(M, 1, T_start, _),
   neg_method_precondition(M, F),
   static(F),
   F.


% Hilfsprädikat zur sequenziellen Abarbeitung der Teilaufgaben einer Methode
% Erledige die erste Teilaufgabe von T1 bis T2 und dann den Rest von T2 bis T3
achieves_subtasks(M, I, T1, T3) :-
    subtask(M, I, S),
    achieves(S, T1, T2),
    achieves_subtasks(M, I + 1, T2, T3).

% Base Case für die Teilaufgaben-Rekursion: Die letzte Teilaufgabe wird erledigt.
achieves_subtasks(M, I, T1, T2) :-
    subtask(M, I, S),
    not subtask(M, I + 1, _), % Es gibt keine nachfolgende Teilaufgabe
    achieves(S, T1, T2).

% Base Case für Methoden ohne Subtasks (leere Methodendeklaration)
achieves_subtasks(M, 1, T, T) :-
    time(T),
    decomposes(M, _),
    not subtask(M, 1, _).


% --- c) Zielbedingung ---

% Test (Constraint): Das Problem ist nur gelöst, wenn alle initialen Aufgaben
% irgendwann erledigt wurden.
:- initial_task(T), not achieves(T, _, _).

% --- Ausgabe ---
#show occurs/2.