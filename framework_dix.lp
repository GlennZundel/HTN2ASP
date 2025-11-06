% ================================================================
% == DOMÄNENUNABHÄNGIGES ASP-PLANUNGS-FRAMEWORK              ==
% ================================================================
% Dieses Framework ist die Grundlage für die Planung.
% Es wird mit den spezifischen Regeln einer Domäne kombiniert.

% Definition der Zeitachse.
time(0..10).

% Erfolgsbedingung: Ein Plan ist gefunden, wenn die initiale Aufgabe 'achieve_goals' erfüllt ist.
% --- 3. Frame-Axiom für Zustandsübergänge ---
% Ein Faktum A bleibt im nächsten Zeitschritt (T+1) wahr,
% wenn es in T wahr war und nicht explizit durch eine Aktion
% entfernt wurde (via 'out_state').
in_state(A, T+1) :-
    time(T),
    atom(A),
    in_state(A, T),
not out_state(A, T+1). 

% --- 4. Erfolgsbedingung für den Plan ---
% Ein Plan ist erfolgreich ('plan_found'), wenn die letzte
% Zielaufgabe 'hn' erreicht (causable) wurde.
% 'hn' und die zugehörigen Zeitvariablen müssen durch das
% spezifische Endziel des Problems ersetzt werden.

% Eine Integritätsbedingung, die sicherstellt, dass nur
% Antwortmengen generiert werden, die einen vollständigen Plan darstellen.

:- not plan_found.
#show causable/3.
% ================================================================
% == Fügen Sie hier die domänenspezifischen Regeln ein:        ==
% == 1. Initialzustand (Trans(S))                             ==
% == 2. Zielaufgaben (Trans(t))                               ==
% == 3. Axiome (Trans(AX))                                    ==
% == 4. Operatoren (Trans(OP))                                ==
% == 5. Methoden (Trans(METH))                                ==
% ================================================================