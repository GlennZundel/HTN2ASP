import sys
import re
from collections import defaultdict

class HDDLParser:
    """Robust HDDL parser using balanced parenthesis matching."""

    def __init__(self, domain_file, problem_file):
        self.domain_content = self._read_file(domain_file)
        self.problem_content = self._read_file(problem_file)
        
        self.types = {}
        self.predicates = {}
        self.tasks = {}
        self.actions = {}
        self.methods = []
        
        self.objects = {}
        self.initial_state = []
        self.goal_state = []  # Neu: Ziele
        self.htn_task = None

    def _read_file(self, file_path):
        """Read file and remove comments."""
        with open(file_path, 'r') as f:
            content = f.read()
        # Remove line comments
        content = re.sub(r';[^\n]*', '', content)
        return content

    def _extract_balanced(self, text, start_pos=0):
        """Extract a balanced S-expression starting at start_pos."""
        depth = 0
        i = start_pos
        start = -1
        
        while i < len(text):
            if text[i] == '(':
                if depth == 0:
                    start = i
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    return text[start:i+1], i+1
            i += 1
        
        return None, len(text)

    def _tokenize_sexp(self, sexp_str):
        """Tokenize an S-expression string."""
        sexp_str = sexp_str.strip()
        if sexp_str.startswith('(') and sexp_str.endswith(')'):
            sexp_str = sexp_str[1:-1].strip()
        
        tokens = []
        current = ""
        depth = 0
        
        for char in sexp_str:
            if char == '(':
                if depth == 0 and current.strip():
                    tokens.append(current.strip())
                    current = ""
                current += char
                depth += 1
            elif char == ')':
                depth -= 1
                current += char
                if depth == 0:
                    tokens.append(current.strip())
                    current = ""
            elif char in ' \t\n' and depth == 0:
                if current.strip():
                    tokens.append(current.strip())
                    current = ""
            else:
                current += char
        
        if current.strip():
            tokens.append(current.strip())
        
        return tokens

    def _parse_typed_list(self, text):
        """Parse 'x1 x2 - type1 y1 - type2' format."""
        tokens = text.split()
        result = []
        current_items = []
        i = 0
        
        while i < len(tokens):
            if tokens[i] == '-' and i + 1 < len(tokens):
                typ = tokens[i + 1].lower()
                for item in current_items:
                    result.append((item, typ))
                current_items = []
                i += 2
            else:
                current_items.append(tokens[i].lower())
                i += 1
        
        for item in current_items:
            result.append((item, 'object'))
        
        return result

    def _parse_condition_list(self, sexp_str):
        """Parse (and (p1) (not (p2)) (p3)) into list of conditions."""
        sexp_str = sexp_str.strip()
        if not sexp_str:
            return []
        
        # Remove outer parens
        if sexp_str.startswith('(') and sexp_str.endswith(')'):
            sexp_str = sexp_str[1:-1].strip()
        
        # Skip 'and'
        if sexp_str.lower().startswith('and'):
            # Find where 'and' ends (next space or paren)
            i = 3
            while i < len(sexp_str) and sexp_str[i] in ' \t\n':
                i += 1
            sexp_str = sexp_str[i:].strip()
        
        result = []
        pos = 0
        
        while pos < len(sexp_str):
            if sexp_str[pos] == '(':
                expr, next_pos = self._extract_balanced(sexp_str, pos)
                if expr:
                    # Remove outer parens
                    inner = expr[1:-1].strip() if expr.startswith('(') and expr.endswith(')') else expr
                    tokens = inner.split()
                    
                    if tokens:
                        if tokens[0].lower() == 'not':
                            # Handle negation
                            if len(tokens) > 1:
                                # Find the negated expression
                                not_pos = len('not')
                                while not_pos < len(inner) and inner[not_pos] in ' \t':
                                    not_pos += 1
                                if not_pos < len(inner) and inner[not_pos] == '(':
                                    negated_expr, _ = self._extract_balanced(inner, not_pos)
                                    negated_atom = self._parse_atom(negated_expr)
                                    result.append(('not', negated_atom))
                                else:
                                    # Simple not case
                                    negated_atom = tuple(t.lower() for t in tokens[1:])
                                    result.append(('not', negated_atom))
                        else:
                            result.append(tuple(t.lower() for t in tokens))
                pos = next_pos
            else:
                pos += 1
        
        return result

    def _parse_atom(self, atom_str):
        """Parse atom like '(pred arg1 arg2)' to tuple."""
        tokens = self._tokenize_sexp(atom_str)
        return tuple(t.lower() for t in tokens)

    def parse(self):
        """Main parsing."""
        self._parse_domain()
        self._parse_problem()

    def _parse_domain(self):
        """Parse domain file."""
        content = self.domain_content
        
        # Parse types
        types_match = re.search(r'\(:types\s+(.*?)\)', content, re.DOTALL | re.IGNORECASE)
        if types_match:
            typed_list = self._parse_typed_list(types_match.group(1))
            for item, parent in typed_list:
                self.types[item] = parent
        
        # Parse predicates
        preds_match = re.search(r'\(:predicates\s+(.*?)\)\s*\(:task', content, re.DOTALL | re.IGNORECASE)
        if preds_match:
            preds_text = preds_match.group(1)
            pos = 0
            while pos < len(preds_text):
                if preds_text[pos] == '(':
                    pred_expr, next_pos = self._extract_balanced(preds_text, pos)
                    if pred_expr:
                        tokens = self._tokenize_sexp(pred_expr)
                        if tokens:
                            name = tokens[0].lower()
                            params_text = ' '.join(tokens[1:])
                            params = self._parse_typed_list(params_text)
                            self.predicates[name] = params
                    pos = next_pos
                else:
                    pos += 1
        
        # Parse tasks
        for task_match in re.finditer(r'\(:task\s+([^\s]+)\s+:parameters\s*\(([^)]*)\)', content, re.IGNORECASE):
            name = task_match.group(1).lower()
            params_text = task_match.group(2)
            params = self._parse_typed_list(params_text)
            self.tasks[name] = params
        
        # Parse actions
        pos = 0
        while True:
            match = re.search(r'\(:action\s+([^\s]+)', content[pos:], re.IGNORECASE)
            if not match:
                break
            
            action_start = pos + match.start()
            action_name = match.group(1).lower()
            
            # Find the action block using balanced extraction
            action_block, next_pos = self._extract_balanced(content, action_start)
            if action_block:
                # Extract parameters
                params = []
                params_match = re.search(r':parameters\s*\(', action_block, re.IGNORECASE)
                if params_match:
                    params_start = action_block.find(':parameters', 0, len(action_block))
                    params_block, _ = self._extract_balanced(action_block, params_start + len(':parameters'))
                    if params_block:
                        params_text = params_block[1:-1] if params_block.startswith('(') else params_block
                        params = self._parse_typed_list(params_text)
                
                # Extract preconditions
                preconditions = []
                pre_match = re.search(r':precondition\s*\(', action_block, re.IGNORECASE)
                if pre_match:
                    pre_start = action_block.find(':precondition', 0, len(action_block))
                    pre_block, _ = self._extract_balanced(action_block, pre_start + len(':precondition'))
                    if pre_block:
                        preconditions = self._parse_condition_list(pre_block)
                
                # Extract effects
                effects = []
                eff_match = re.search(r':effect\s*\(', action_block, re.IGNORECASE)
                if eff_match:
                    eff_start = action_block.find(':effect', 0, len(action_block))
                    eff_block, _ = self._extract_balanced(action_block, eff_start + len(':effect'))
                    if eff_block:
                        effects = self._parse_condition_list(eff_block)
                
                self.actions[action_name] = {
                    'params': params,
                    'pre': preconditions,
                    'eff': effects
                }
            
            pos = action_start + len(action_block) if action_block else pos + 1
        
        # Parse methods
        pos = 0
        while True:
            match = re.search(r'\(:method\s+([^\s]+)', content[pos:], re.IGNORECASE)
            if not match:
                break
            
            method_start = pos + match.start()
            method_name = match.group(1).lower()
            
            method_block, next_pos = self._extract_balanced(content, method_start)
            if method_block:
                # Extract parameters
                params = []
                params_match = re.search(r':parameters\s*\(', method_block, re.IGNORECASE)
                if params_match:
                    params_start = method_block.find(':parameters', 0, len(method_block))
                    params_block, _ = self._extract_balanced(method_block, params_start + len(':parameters'))
                    if params_block:
                        params_text = params_block[1:-1] if params_block.startswith('(') else params_block
                        params = self._parse_typed_list(params_text)
                
                # Extract task
                task = None
                task_match = re.search(r':task\s*\(', method_block, re.IGNORECASE)
                if task_match:
                    task_start = method_block.find(':task', 0, len(method_block))
                    task_block, _ = self._extract_balanced(method_block, task_start + len(':task'))
                    if task_block:
                        task = self._parse_atom(task_block)
                
                # Extract preconditions
                preconditions = []
                pre_match = re.search(r':precondition\s*\(', method_block, re.IGNORECASE)
                if pre_match:
                    pre_start = method_block.find(':precondition', 0, len(method_block))
                    pre_block, _ = self._extract_balanced(method_block, pre_start + len(':precondition'))
                    if pre_block:
                        preconditions = self._parse_condition_list(pre_block)
                
                # Extract subtasks
                subtasks = []
                subtasks_match = re.search(r':(?:ordered-tasks|ordered-subtasks)\s*\(', method_block, re.IGNORECASE)
                if subtasks_match:
                    keyword = ':ordered-tasks' if ':ordered-tasks' in method_block.lower() else ':ordered-subtasks'
                    subtasks_start = method_block.lower().find(keyword.lower(), 0, len(method_block))
                    subtasks_block, _ = self._extract_balanced(method_block, subtasks_start + len(keyword))
                    if subtasks_block:
                        # Parse subtasks
                        subtasks_text = subtasks_block[1:-1] if subtasks_block.startswith('(') and subtasks_block.endswith(')') else subtasks_block
                        subtasks_text = subtasks_text.strip()
                        
                        # Check for empty or just 'and'
                        if subtasks_text and subtasks_text.lower() != 'and':
                            # Skip 'and' if present
                            if subtasks_text.lower().startswith('and'):
                                i = 3
                                while i < len(subtasks_text) and subtasks_text[i] in ' \t\n':
                                    i += 1
                                subtasks_text = subtasks_text[i:].strip()
                            
                            # Extract subtasks
                            if subtasks_text:
                                st_pos = 0
                                while st_pos < len(subtasks_text):
                                    if subtasks_text[st_pos] == '(':
                                        # Parenthesized subtask  
                                        st_expr, st_next = self._extract_balanced(subtasks_text, st_pos)
                                        if st_expr:
                                            subtasks.append(self._parse_atom(st_expr))
                                        st_pos = st_next
                                    elif subtasks_text[st_pos] not in ' \t\n':
                                        # Non-parenthesized subtask - treat the entire remaining text as one subtask
                                        remaining = subtasks_text[st_pos:].strip()
                                        tokens = remaining.split()
                                        if tokens:
                                            subtasks.append(tuple(t.lower() for t in tokens))
                                        break  # Done parsing subtasks
                                    else:
                                        st_pos += 1
                
                self.methods.append({
                    'name': method_name,
                    'params': params,
                    'task': task,
                    'preconditions': preconditions,
                    'subtasks': subtasks
                })
            
            pos = method_start + len(method_block) if method_block else pos + 1

    def _parse_problem(self):
        """Parse problem file."""
        content = self.problem_content
        
        # Parse objects
        objects_match = re.search(r'\(:objects\s+(.*?)\)', content, re.DOTALL | re.IGNORECASE)
        if objects_match:
            typed_list = self._parse_typed_list(objects_match.group(1))
            for obj, typ in typed_list:
                self.objects[obj] = typ
        
        # Parse init
        init_match = re.search(r'\(:init\s+(.*?)(?:\(:goal|\(:htn|\)$)', content, re.DOTALL | re.IGNORECASE)
        if init_match:
            init_text = init_match.group(1)
            pos = 0
            while pos < len(init_text):
                if init_text[pos] == '(':
                    atom_expr, next_pos = self._extract_balanced(init_text, pos)
                    if atom_expr:
                        self.initial_state.append(self._parse_atom(atom_expr))
                    pos = next_pos
                else:
                    pos += 1
        
        # Parse HTN
        htn_match = re.search(r'\(:htn\s+(.*?)\)\s*\(:(?:goal|init|\))', content, re.DOTALL | re.IGNORECASE)
        if not htn_match:
            htn_match = re.search(r'\(:htn\s+(.*)\)', content, re.DOTALL | re.IGNORECASE)
        if htn_match:
            htn_text = htn_match.group(1)
            task_match = re.search(r'\(task\d*\s+\(([^)]+)\)', htn_text, re.IGNORECASE)
            if task_match:
                task_tokens = task_match.group(1).split()
                self.htn_task = tuple(t.lower() for t in task_tokens)
        
        # Parse goals
        goal_match = re.search(r'\(:goal\s+', content, re.IGNORECASE)
        if goal_match:
            goal_start = goal_match.start()
            goal_block, _ = self._extract_balanced(content, goal_start)
            if goal_block:
                # Remove (:goal and outer parens
                goal_text = goal_block[1:-1] if goal_block.startswith('(') and goal_block.endswith(')') else goal_block
                # Find where :goal ends
                goal_keyword_end = goal_text.lower().find(':goal') + len(':goal')
                goal_content = goal_text[goal_keyword_end:].strip()
                if goal_content:
                    self.goal_state = self._parse_condition_list(goal_content)


class ASPTranslator:
    """Translate HDDL to ASP."""

    def __init__(self, parsed_data):
        self.data = parsed_data
        self.method_groups = self._group_methods()

    def _group_methods(self):
        """Group methods by task."""
        groups = defaultdict(list)
        for method in self.data.methods:
            if method['task']:
                task_name = method['task'][0]
                groups[task_name].append(method)
        return groups

    def _fmt_term(self, term):
        """Format term (?x -> X, a -> a)."""
        if term.startswith('?'):
            return term[1:].upper()
        return term

    def _fmt_atom(self, atom):
        """Format atom as ASP."""
        if not atom:
            return ""
        name = atom[0].replace('-', '_')  # Replace hyphens with underscores
        params = [self._fmt_term(p) for p in atom[1:]]
        if not params:
            return name
        return f"{name}({', '.join(params)})"

    def translate(self):
        """Generate ASP code."""
        lines = []
        
        lines.append("% ================================================================")
        lines.append("% == ASP-Übersetzung für die Roboter-Domäne                    ==")
        lines.append("% == Basierend auf der Methodik von Dix, Kuter & Nau (2005)   ==")
        lines.append("% ================================================================")
        lines.append("")
        
        # 1. Constants
        lines.append("% ################################################################")
        lines.append("% ## 1. KONSTANTEN UND ATOM-DEKLARATIONEN (aus Domain & Problem)##")
        lines.append("% ################################################################")
        lines.append("")
        
        type_objs = defaultdict(list)
        for obj, typ in self.data.objects.items():
            type_objs[typ].append(obj)
        
        for typ in sorted(type_objs.keys()):
            objs = type_objs[typ]
            lines.append(f"{typ}({'; '.join(objs)}).")
        
        lines.append("")
        lines.append("% --- Deklaration aller möglichen Atome für das Grounding ---")
        lines.append("")
        
        for pred_name in sorted(self.data.predicates.keys()):
            params = self.data.predicates[pred_name]
            if not params:
                lines.append(f"atom({pred_name}).")
            else:
                vars = [self._fmt_term(v) for v, _ in params]
                types = [t for _, t in params]
                type_conds = [f"{types[i]}({vars[i]})" for i in range(len(params))]
                atom_str = f"{pred_name}({', '.join(vars)})"
                lines.append(f"atom({atom_str}) :- {', '.join(type_conds)}.")
        
        lines.append("")
        lines.append("")
        
        # 2. Initial state
        lines.append("% ################################################################")
        lines.append("% ## 2. INITIALZUSTAND - Trans(S)                               ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("")
        
        for atom in self.data.initial_state:
            lines.append(f"in_state({self._fmt_atom(atom)}, 0).")
        
        lines.append("")
        lines.append("")
        
        # 3. Goal task
        lines.append("% ################################################################")
        lines.append("% ## 3. ZIELAUFGABE(N) - Trans(t)                               ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("")
        
        if self.data.htn_task:
            task_name = self.data.htn_task[0].replace('-', '_')
            lines.append(f"% Die initiale Aufgabe zur Zeit 0 ist '{self.data.htn_task[0]}'.")
            lines.append(f"taskTBA({task_name}, 0).")
        
        lines.append("")
        lines.append("")
        
        # 4. Actions
        lines.append("% ################################################################")
        lines.append("% ## 4. OPERATOREN (Aktionen) - Trans(OP)                       ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("")
        
        for action_name in sorted(self.data.actions.keys()):
            action_data = self.data.actions[action_name]
            lines.append(f"% --- Operator '{action_name}' ---")
            
            param_vars = [v for v, _ in action_data['params']]
            task_head = self._fmt_atom(tuple([action_name] + param_vars))
            
            for eff in action_data['eff']:
                if isinstance(eff, tuple) and eff[0] == 'not':
                    atom_str = self._fmt_atom(eff[1])
                    lines.append(f"out_state({atom_str}, T+1)   :- taskTBA({task_head}, T).")
                else:
                    atom_str = self._fmt_atom(eff)
                    lines.append(f"in_state({atom_str}, T+1) :- taskTBA({task_head}, T).")
            
            lines.append("")
        
        lines.append("")
        
        # 5. Methods
        lines.append("% ################################################################")
        lines.append("% ## 5. METHODEN - Trans(METH)                                  ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("")
        lines.append("% --- 5.1 Übersetzung von \"abstrakten\" zu \"primitiven\" Aufgaben ---")
        lines.append("")
        lines.append("")
        
        # Simple methods (abstract to primitive)
        for method in sorted(self.data.methods, key=lambda m: m['name']):
            if method['name'].startswith('newmethod'):
                self._translate_simple_method(method, lines)
        
        lines.append("")
        lines.append("% --- 5.2 Übersetzung der Kontrollfluss-Methoden ---")
        lines.append("")
        
        # Complex methods
        for task_name in sorted(self.method_groups.keys()):
            if not task_name.endswith('_abstract'):
                self._translate_complex_methods(task_name, lines)
        
        lines.append("")
        
        # 6. Causable
        lines.append("% ################################################################")
        lines.append("% ## 6. CAUSABLE DEFINITION FÜR PRIMITIVE AUFGABEN              ##")
        lines.append("% ################################################################")
        lines.append("")
        
        for action_name in sorted(self.data.actions.keys()):
            action_data = self.data.actions[action_name]
            param_vars = [v for v, _ in action_data['params']]
            task_head = self._fmt_atom(tuple([action_name] + param_vars))
            lines.append(f"causable({task_head}, T, T+1)   :- taskTBA({task_head}, T).")
        
        lines.append("")
        lines.append("")
        
        # 7. Deklaration primitiver Tasks (für Constraints)
        lines.append("% ################################################################")
        lines.append("% ## 7. DEKLARATION PRIMITIVER TASKS                            ##")
        lines.append("% ################################################################")
        lines.append("")
        
        for action_name in sorted(self.data.actions.keys()):
            action_data = self.data.actions[action_name]
            param_vars = [v for v, _ in action_data['params']]
            param_types = [t for _, t in action_data['params']]
            
            # Format task head
            task_head = self._fmt_atom(tuple([action_name] + param_vars))
            
            # Generate type constraints
            if param_vars:
                type_constraints = [f"{param_types[i]}({self._fmt_term(param_vars[i])})" 
                                  for i in range(len(param_vars))]
                lines.append(f"primitive_task({task_head}) :- {', '.join(type_constraints)}.")
            else:
                lines.append(f"primitive_task({task_head}).")
        
        lines.append("")
        lines.append("")
        
        # 8. Trans(G) - Goal Translation
        lines.append("% ################################################################")
        lines.append("% ## 8. ZIEL-ÜBERSETZUNG - Trans(G)                             ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("% Die Ziele aus der :goal Sektion des Problems müssen erfüllt sein")
        lines.append("")
        
        if self.data.goal_state:
            lines.append("% Für jedes Ziel: Es muss zu IRGENDEINEM Zeitpunkt erfüllt sein")
            for i, goal in enumerate(self.data.goal_state):
                if isinstance(goal, tuple) and goal[0] == 'not':
                    atom_str = self._fmt_atom(goal[1])
                    lines.append(f"goal_{i}_satisfied :- not in_state({atom_str}, T), time(T).")
                else:
                    atom_str = self._fmt_atom(goal)
                    lines.append(f"goal_{i}_satisfied :- in_state({atom_str}, T), time(T).")
            
            # Alle Ziele müssen erfüllt sein
            goal_conditions = [f"goal_{i}_satisfied" for i in range(len(self.data.goal_state))]
            lines.append(f"goal_satisfied :- {', '.join(goal_conditions)}.")
        else:
            lines.append("% Keine expliziten Ziele definiert")
            lines.append("goal_satisfied.")
        
        lines.append("")
        lines.append("")
        
        # 9. Trans(⊥) - Successful Termination
        lines.append("% ################################################################")
        lines.append("% ## 9. ERFOLGREICHE TERMINIERUNG - Trans(⊥)                    ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("% Ein Plan ist erfolgreich, wenn:")
        lines.append("% 1. Die initiale Aufgabe causable ist")
        lines.append("% 2. Alle Ziele im Endzustand erfüllt sind")
        lines.append("")
        
        if self.data.htn_task:
            task_name = self.data.htn_task[0].replace('-', '_')
            if len(self.data.htn_task) > 1:
                task_params = ', '.join(self._fmt_term(p) for p in self.data.htn_task[1:])
                lines.append(f"plan_found :- causable({task_name}({task_params}), 0, T_final), goal_satisfied, time(T_final).")
            else:
                lines.append(f"plan_found :- causable({task_name}, 0, T_final), goal_satisfied, time(T_final).")
        
        lines.append("")
        lines.append("% Integrity Constraint: Es muss einen Plan geben")
        lines.append(":- not plan_found.")
        
        lines.append("")
        lines.append("")
        
        # 10. Constraints
        lines.append("% ################################################################")
        lines.append("% ## 10. CONSTRAINTS                                             ##")
        lines.append("% ################################################################")
        lines.append("")
        lines.append("% Constraint 1: Nur eine primitive Aktion pro Zeitschritt")
        lines.append("% (Verhindert gleichzeitige Ausführung mehrerer Aktionen)")
        lines.append(":- taskTBA(A1, T), taskTBA(A2, T), A1 != A2,")
        lines.append("   primitive_task(A1), primitive_task(A2), time(T).")
        
        return '\n'.join(lines)

    def _translate_simple_method(self, method, lines):
        """Translate abstract-to-primitive method."""
        task_name = method['task'][0] if method['task'] else ''
        task_name_asp = task_name.replace('-', '_')
        method_pred = f"method_{task_name_asp}"
        
        # Method head
        if method['task'] and len(method['task']) > 1:
            task_params = [self._fmt_term(p) for p in method['task'][1:]]
            lines.append(f"% Methode für '{task_name}' -> '{method['subtasks'][0][0] if method['subtasks'] else ''}'")
            lines.append(f"{method_pred}({', '.join(task_params)}, T) :- taskTBA({task_name_asp}({', '.join(task_params)}), T).")
        else:
            lines.append(f"% Methode für '{task_name}' -> '{method['subtasks'][0][0] if method['subtasks'] else ''}'")
            lines.append(f"{method_pred}(T) :- taskTBA({task_name_asp}, T).")
        
        if not method['subtasks']:
            lines.append("")
            return
        
        # Get action
        subtask = method['subtasks'][0]
        action_name = subtask[0]
        action_data = self.data.actions.get(action_name, {})
        
        if not action_data:
            lines.append("")
            return
        
        subtask_atom = self._fmt_atom(subtask)
        
        # taskTBA rule with preconditions
        preconditions = []
        for pre in action_data.get('pre', []):
            if isinstance(pre, tuple) and pre[0] == 'not':
                preconditions.append(f"not in_state({self._fmt_atom(pre[1])}, T)")
            else:
                preconditions.append(f"in_state({self._fmt_atom(pre)}, T)")
        
        if method['task'] and len(method['task']) > 1:
            method_call = f"{method_pred}({', '.join([self._fmt_term(p) for p in method['task'][1:]])}, T)"
        else:
            method_call = f"{method_pred}(T)"
        
        lines.append(f"taskTBA({subtask_atom}, T) :-")
        lines.append(f"    {method_call},")
        for i, pre in enumerate(preconditions):
            suffix = "." if i == len(preconditions) - 1 else ","
            comment = f" % Preconditions der '{action_name}'-Aktion" if i == len(preconditions) - 1 else ""
            lines.append(f"    {pre}{suffix}{comment}")
        
        # Causable rule
        lines.append(f"causable({self._fmt_atom(method['task'])}, T_start, T_end) :-")
        
        cond_lines = []
        
        # Method call
        if method['task'] and len(method['task']) > 1:
            cond_lines.append(f"{method_pred}({', '.join([self._fmt_term(p) for p in method['task'][1:]])}, T_start)")
        else:
            cond_lines.append(f"{method_pred}(T_start)")
        
        # Type constraints
        for v, t in method['params']:
            cond_lines.append(f"{t}({self._fmt_term(v)})")
        
        # Preconditions
        for pre in action_data.get('pre', []):
            if isinstance(pre, tuple) and pre[0] == 'not':
                cond_lines.append(f"not in_state({self._fmt_atom(pre[1])}, T_start)")
            else:
                cond_lines.append(f"in_state({self._fmt_atom(pre)}, T_start)")
        
        # Causable of action
        cond_lines.append(f"causable({subtask_atom}, T_start, T_end)")
        
        for i, cond in enumerate(cond_lines):
            suffix = "." if i == len(cond_lines) - 1 else ","
            lines.append(f"    {cond}{suffix}")
        
        lines.append("")

    def _translate_complex_methods(self, task_name, lines):
        """Translate complex task methods."""
        methods = self.method_groups.get(task_name, [])
        if not methods:
            return
        
        task_name_asp = task_name.replace('-', '_')
        
        lines.append(f"% --- Methoden für '{task_name}' ---")
        lines.append("% Nicht-deterministische Auswahl einer der möglichen Methoden")
        
        # Generate choice rules
        method_preds = [f"method_{m['name'].replace('-', '_')}" for m in methods]
        for i, mpred in enumerate(method_preds):
            others = [f"not {mp}(T)" for j, mp in enumerate(method_preds) if j != i]
            if others:
                lines.append(f"{mpred}(T) :- taskTBA({task_name_asp}, T), {', '.join(others)}.")
            else:
                lines.append(f"{mpred}(T) :- taskTBA({task_name_asp}, T).")
        
        lines.append("")
        
        # Translate each method
        for method in methods:
            self._translate_single_method(method, task_name_asp, lines)

    def _translate_single_method(self, method, task_name_asp, lines):
        """Translate one complex method."""
        method_name = method['name']
        method_pred = f"method_{method_name.replace('-', '_')}"
        
        lines.append(f"% Methode '{method_name}'")
        
        subtasks = method['subtasks']
        preconditions = method['preconditions']
        params = method['params']
        
        if not subtasks:
            # Base case
            lines.append(f"causable({task_name_asp}, T, T) :- {method_pred}(T).")
            lines.append("")
            return
        
        # Build condition list
        conds = []
        if params:
            for v, t in params:
                conds.append(f"{t}({self._fmt_term(v)})")
        
        for pre in preconditions:
            if isinstance(pre, tuple) and pre[0] == 'not':
                conds.append(f"not in_state({self._fmt_atom(pre[1])}, T)")
            else:
                conds.append(f"in_state({self._fmt_atom(pre)}, T)")
        
        cond_str = f", {', '.join(conds)}" if conds else ""
        
        if len(subtasks) == 1:
            # Single subtask
            subtask_atom = self._fmt_atom(subtasks[0])
            lines.append(f"taskTBA({subtask_atom}, T) :-")
            lines.append(f"    {method_pred}(T){cond_str}.")
            
            # Check if recursive
            if subtasks[0][0].replace('-', '_') != task_name_asp:
                lines.append(f"causable({task_name_asp}, T, T2) :-")
                lines.append(f"    {method_pred}(T){cond_str},")
                lines.append(f"    causable({subtask_atom}, T, T2).")
            
        elif len(subtasks) == 2:
            # Two subtasks
            st1_atom = self._fmt_atom(subtasks[0])
            st2_atom = self._fmt_atom(subtasks[1])
            
            lines.append(f"taskTBA({st1_atom}, T) :-")
            lines.append(f"    {method_pred}(T){cond_str}.")
            
            lines.append(f"taskTBA({st2_atom}, T2) :-")
            lines.append(f"    {method_pred}(T){cond_str},")
            lines.append(f"    causable({st1_atom}, T, T2), T2 >= T.")
            
            lines.append(f"causable({task_name_asp}, T, T3) :-")
            lines.append(f"    {method_pred}(T){cond_str},")
            lines.append(f"    causable({st2_atom}, T2, T3), T3 >= T2, causable({st1_atom}, T, T2).")
        
        lines.append("")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python hddl_to_lp_translator.py <domain.hddl> <problem.hddl> <output.lp>")
        sys.exit(1)

    domain_file = sys.argv[1]
    problem_file = sys.argv[2]
    output_file = sys.argv[3]

    try:
        parser = HDDLParser(domain_file, problem_file)
        parser.parse()
        
        translator = ASPTranslator(parser)
        asp_program = translator.translate()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(asp_program)
            
        print(f"Successfully translated to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()