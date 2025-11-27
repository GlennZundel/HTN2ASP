import sys
import re
import os
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
        self.goal_state = []  
        self.htn_tasks = []

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

        # Handle single atom case (no more parentheses)
        if '(' not in sexp_str:
            # Single atom case, e.g., "factory-at ?f ?l" or "not factory-at ?f ?l"
            tokens = sexp_str.split()
            if tokens:
                if tokens[0].lower() == 'not':
                    # Handle negation without parens: "not factory-at ?f ?l"
                    negated_atom = tuple(t.lower() for t in tokens[1:])
                    return [('not', negated_atom)]
                else:
                    # Simple positive atom
                    return [tuple(t.lower() for t in tokens)]
            return []

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
            """Parse problem file robustly using balanced extraction."""
            content = self.problem_content
            
            # 1. Parse objects
            obj_match = re.search(r'\(:objects\s', content, re.IGNORECASE)
            if obj_match:
                start = obj_match.start()
                block, _ = self._extract_balanced(content, start)
                if block:
                    # Inhalt ohne (:objects und )
                    # Wir suchen die erste Klammer nach :objects für den Fall von (:objects (a - type))
                    # oder nehmen einfach den Text nach dem Keyword
                    keyword_end = block.lower().find(':objects') + len(':objects')
                    inner = block[keyword_end:-1].strip()
                    typed_list = self._parse_typed_list(inner)
                    for obj, typ in typed_list:
                        self.objects[obj] = typ

            # 2. Parse init
            init_match = re.search(r'\(:init\s', content, re.IGNORECASE)
            if init_match:
                start = init_match.start()
                block, _ = self._extract_balanced(content, start)
                if block:
                    keyword_end = block.lower().find(':init') + len(':init')
                    inner = block[keyword_end:-1].strip()
                    pos = 0
                    while pos < len(inner):
                        if inner[pos] == '(':
                            atom_expr, next_pos = self._extract_balanced(inner, pos)
                            if atom_expr:
                                self.initial_state.append(self._parse_atom(atom_expr))
                            pos = next_pos
                        else:
                            pos += 1

            # 3. Parse HTN
            htn_match = re.search(r'\(:htn\s', content, re.IGNORECASE)
            if htn_match:
                start = htn_match.start()
                block, _ = self._extract_balanced(content, start)
                if block:
                    # Suche nach der Task-Liste (:ordered-tasks, :subtasks, etc.)
                    st_match = re.search(r':(?:ordered-tasks|ordered-subtasks|tasks|subtasks)\s', block, re.IGNORECASE)
                    if st_match:
                        # Finde den Start der Liste (erste Klammer nach dem Keyword)
                        list_start = block.find('(', st_match.end())
                        if list_start != -1:
                            task_list_expr, _ = self._extract_balanced(block, list_start)
                            if task_list_expr:
                                inner_content = task_list_expr
                                
                                # Robustes Entfernen von (and ...)
                                # Prüfen, ob der Block mit (and beginnt (case insensitive, ignoring whitespace)
                                if re.match(r'\(\s*and\s', inner_content, re.IGNORECASE):
                                    # Entferne äußere Klammern
                                    inner_content = inner_content[1:-1].strip()
                                    # Entferne das 'and' am Anfang
                                    inner_content = re.sub(r'^and\s+', '', inner_content, flags=re.IGNORECASE).strip()
                                elif inner_content.startswith('(') and inner_content.endswith(')'):
                                    # Entferne äußere Klammern des Listen-Wrappers
                                    # ABER VORSICHT: Wenn es (task1 arg) ist (einzelne Task), dürfen wir nicht strippen,
                                    # wenn es ((task1) (task2)) ist, müssen wir strippen.
                                    
                                    # Check: Ist das erste Element im Inneren wieder eine Klammer?
                                    temp_inner = inner_content[1:-1].strip()
                                    if temp_inner.startswith('('):
                                        inner_content = temp_inner
                                    # Sonst (bei einzelner Task wie (do_stuff a b)) lassen wir die Klammern dran,
                                    # damit der Loop unten sie als EINE Task findet.

                                # Iteriere durch die Tasks
                                pos = 0
                                while pos < len(inner_content):
                                    if inner_content[pos] == '(':
                                        task_expr, next_pos = self._extract_balanced(inner_content, pos)
                                        if task_expr:
                                            # task_expr ist z.B. (task0 (achieve-goals)) ODER (move a b)
                                            
                                            # Tokenize um Struktur zu prüfen
                                            tokens = self._tokenize_sexp(task_expr)
                                            
                                            # UNWRAP LOGIC:
                                            # Wenn wir genau 2 Tokens haben und das zweite mit '(' beginnt,
                                            # ist es wahrscheinlich ein Label: (label (actual_task ...))
                                            if len(tokens) == 2 and tokens[1].startswith('('):
                                                # Wir parsen den inneren Teil als Task
                                                inner_task_str = tokens[1]
                                                self.htn_tasks.append(self._parse_atom(inner_task_str))
                                            else:
                                                # Standard Task: (name arg1 arg2)
                                                self.htn_tasks.append(tuple(t.lower() for t in tokens))
                                                
                                        pos = next_pos
                                    else:
                                        pos += 1

            # 4. Parse goals
            goal_match = re.search(r'\(:goal\s', content, re.IGNORECASE)
            if goal_match:
                start = goal_match.start()
                block, _ = self._extract_balanced(content, start)
                if block:
                    keyword_end = block.lower().find(':goal') + len(':goal')
                    inner = block[keyword_end:-1].strip()
                    self.goal_state = self._parse_condition_list(inner)


class ASPTranslator:
    """Translate HDDL to ASP."""

    def __init__(self, parsed_data):
        self.data = parsed_data
        self.method_groups = self._group_methods()

    def _preprocess_actions_with_preconditions(self):
        """
        Für jede action mit preconditions: erstelle compound task wrapper + methode.
        Returns: dict mapping original action name to wrapper task name
        """
        action_wrappers = {}
        new_methods = []
        
        for action_name, action in self.data.actions.items():
            if action['pre']:  # Hat preconditions
                # Erstelle compound task wrapper
                wrapper_name = f"{action_name}_wrapper"
                wrapper_params = action['params']
                
                # Registriere als compound task
                self.data.tasks[wrapper_name] = wrapper_params
                
                # Erstelle wrapper task atom
                param_names = tuple(p[0] for p in wrapper_params)
                wrapper_task = (wrapper_name,) + param_names
                
                # Erstelle subtask (die originale primitive task)
                original_subtask = (action_name,) + param_names
                
                # Erstelle Methode die wrapper zerlegt
                method = {
                    'name': f"m_{wrapper_name}",
                    'params': wrapper_params,
                    'task': wrapper_task,
                    'preconditions': action['pre'],  # Preconditions der action
                    'subtasks': [original_subtask]   # Nur die primitive task
                }
                
                new_methods.append(method)
                action_wrappers[action_name] = wrapper_name
        
        # Füge neue Methoden hinzu
        self.data.methods.extend(new_methods)
        
        # Update method_groups
        self.method_groups = self._group_methods()
        
        return action_wrappers

    def _replace_action_references_in_methods(self, action_wrappers):
        """
        Ersetze in allen Methoden die Referenzen auf primitive tasks mit preconditions
        durch ihre wrapper compound tasks.
        """
        for method in self.data.methods:
            # Skip die neu erstellten wrapper-methoden
            if method['name'].startswith('m_') and method['name'].endswith('_wrapper'):
                continue
            
            # Ersetze in subtasks
            new_subtasks = []
            for subtask in method['subtasks']:
                task_name = subtask[0]
                if task_name in action_wrappers:
                    # Ersetze durch wrapper
                    wrapper_name = action_wrappers[task_name]
                    new_subtask = (wrapper_name,) + subtask[1:]
                    new_subtasks.append(new_subtask)
                else:
                    new_subtasks.append(subtask)
            
            method['subtasks'] = new_subtasks

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

    def _get_time_var(self, n):
        """Generate time variable name. Returns 'T' for n<=1, 'T{n}' for n>1."""
        if n <= 1:
            return "T"
        return f"T{n}"

    def _get_variables_from_conditions(self, conditions):
        """Extract all variables from condition atoms."""
        variables = set()
        for cond in conditions:
            if isinstance(cond, tuple) and cond[0] == 'not':
                # Negated condition
                atom = cond[1]
            else:
                atom = cond
            
            # Extract variables (terms starting with ?)
            for term in atom[1:] if len(atom) > 1 else []:
                if term.startswith('?'):
                    variables.add(term)
        return variables

    def _get_typing_constraints(self, variables, param_list):
        """Generate typing constraints for variables based on parameter list."""
        # Build a map from variable name to type
        type_map = {}
        for var, typ in param_list:
            type_map[var] = typ
        
        # Generate typing constraints
        typing_clauses = []
        for var in variables:
            var_lower = var.lower()
            if var_lower in type_map:
                typ = type_map[var_lower]
                # Only add if type is not 'object' (default)
                if typ != 'object':
                    typing_clauses.append(f"{typ}({self._fmt_term(var)})")
        
        return typing_clauses

    def _get_task_typing_constraints(self, task_atom):
        """Generate typing constraints for a task's parameters."""
        if not task_atom or len(task_atom) < 2:
            return []
        
        task_name = task_atom[0]
        task_params = task_atom[1:]
        
        # Get parameter types from actions or tasks
        param_types = None
        if task_name in self.data.actions:
            param_types = self.data.actions[task_name]['params']
        elif task_name in self.data.tasks:
            param_types = self.data.tasks[task_name]
        
        if not param_types:
            return []
        
        # Generate typing constraints
        typing_clauses = []
        for i, param in enumerate(task_params):
            if param.startswith('?') and i < len(param_types):
                typ = param_types[i][1]
                if typ != 'object':
                    typing_clauses.append(f"{typ}({self._fmt_term(param)})")
        
        return typing_clauses

    def _fmt_method_head(self, method, time_var):
        """Format method head with parameters."""
        name = method['name'].replace('-', '_')
        params = [self._fmt_term(p[0]) for p in method['params']]
        if not params:
            return f"{name}({time_var})"
        return f"{name}({', '.join(params)}, {time_var})"

    def translate_domain(self):
        """Translate domain to ASP."""
        action_wrappers = self._preprocess_actions_with_preconditions()
        self._replace_action_references_in_methods(action_wrappers)

        rules = []
        
        # Header comment
        rules.append("% Primitive tasks (actions)")
        rules.append("")
        
        # Translate primitive tasks (actions)
        for action_name, action in self.data.actions.items():
            # Build action atom
            param_names = tuple(p[0] for p in action['params'])
            action_atom_tuple = (action_name,) + param_names
            action_atom = self._fmt_atom(action_atom_tuple)
            
            # Get typing constraints for action parameters
            action_typing = self._get_task_typing_constraints(action_atom_tuple)
            
            # Build body for causable rule
            body_parts = [f"taskTBA({action_atom}, T)"] + action_typing
            body = ", ".join(body_parts)
            
            # causable rule for primitive task
            causable_rule = f"causable({action_atom}, T, T+1) :- {body}."
            rules.append(causable_rule)
            
            # Effects
            for eff in action['eff']:
                if isinstance(eff, tuple) and eff[0] == 'not':
                    # Delete effect
                    eff_atom = self._fmt_atom(eff[1])
                    delete_rule = f"out_state({eff_atom}, T+1) :- taskTBA({action_atom}, T)."
                    rules.append(delete_rule)
                else:
                    # Add effect
                    eff_atom = self._fmt_atom(eff)
                    add_rule = f"in_state({eff_atom}, T+1) :- taskTBA({action_atom}, T)."
                    rules.append(add_rule)
            
            rules.append("")
        
        # Translate methods
        rules.append("% Method selection rules")
        rules.append("")
        
        for task_name, methods in self.method_groups.items():
            for method in methods:
                method_head = self._fmt_method_head(method, "T")
                task_atom = self._fmt_atom(method['task'])

                # Get typing constraints for the task parameters
                task_typing = self._get_task_typing_constraints(method['task'])

                # Get typing constraints for the method parameters (all of them)
                method_typing = self._get_typing_constraints(
                    [p[0] for p in method['params']],
                    method['params']
                )

                # Generate "not" clauses for other methods
                other_methods_clauses = []
                for m in methods:
                    if m['name'] != method['name']:
                        m_name = m['name'].replace('-', '_')
                        # Count params
                        n_params = len(m['params'])
                        if n_params > 0:
                            anon_args = ", ".join(["_"] * n_params)
                            other_methods_clauses.append(f"not {m_name}({anon_args}, T)")
                        else:
                            other_methods_clauses.append(f"not {m_name}(T)")

                # Build body parts WITHOUT preconditions
                body_parts = [f"taskTBA({task_atom}, T)"]
                # Add typing constraints (merge task and method typing, remove duplicates)
                all_typing = sorted(list(set(task_typing + method_typing)))
                body_parts.extend(all_typing)
                body_parts.extend(other_methods_clauses)

                # Simple rule without cardinality constraints
                body = ", ".join(body_parts)
                method_rule = f"{method_head} :- {body}."
                rules.append(method_rule)
            rules.append("")
        
        # Translate subtasks for each method
        rules.append("% Subtask rules")
        rules.append("")
        
        for task_name, methods in self.method_groups.items():
            for method in methods:
                method_head = self._fmt_method_head(method, "T")
                subtasks = method['subtasks']
                preconditions = method['preconditions']

                # Build precondition clauses
                precond_clauses = []
                for precond in preconditions:
                    if isinstance(precond, tuple) and precond[0] == 'not':
                        precond_clauses.append(f"not in_state({self._fmt_atom(precond[1])}, T)")
                    else:
                        precond_clauses.append(f"in_state({self._fmt_atom(precond)}, T)")

                # Get typing constraints for variables in preconditions
                precond_vars = self._get_variables_from_conditions(preconditions)
                typing_clauses = self._get_typing_constraints(precond_vars, method['params'])

                # Generate subtask rules
                for i, subtask in enumerate(subtasks):
                    subtask_atom = self._fmt_atom(subtask)

                    # Get typing constraints for the subtask's parameters
                    task_typing_clauses = self._get_task_typing_constraints(subtask)

                    # Build rule body (with preconditions)
                    body_parts = [f"{method_head}"] + precond_clauses + typing_clauses
                    
                    # Add causable constraint for immediately previous subtask
                    if i > 0:
                        prev_subtask = self._fmt_atom(subtasks[i-1])
                        prev_time = "T" if i == 1 else f"T{i}"
                        curr_time = f"T{i+1}"
                        body_parts.append(f"causable({prev_subtask}, {prev_time}, {curr_time})")
                        body_parts.append(f"{curr_time} >= {prev_time}")
                    
                    # Determine time variable for this subtask
                    if i == 0:
                        time_var = "T"
                    else:
                        time_var = f"T{i+1}"
                    
                    # Apply cardinality constraints for primitive tasks only
                    # Remove duplicate typing constraints
                    existing_constraints = set(body_parts)
                    unique_task_typing = [tc for tc in task_typing_clauses if tc not in existing_constraints]
                    body_with_types = body_parts + unique_task_typing
                    body = ", ".join(body_with_types)

                    # Check if this subtask is a primitive task (action) or compound task
                    subtask_name = subtask[0]
                    is_primitive = subtask_name in self.data.actions

                    if is_primitive:
                        # Cardinality constraint for primitive tasks
                        subtask_rule = f"0 {{ taskTBA({subtask_atom}, {time_var}) : {body} }} 1 :- time(T)."
                    else:
                        # Regular rule for compound tasks
                        subtask_rule = f"taskTBA({subtask_atom}, {time_var}) :- {body}."

                    rules.append(subtask_rule)
                
                rules.append("")
        
        # Causable rules for compound tasks
        rules.append("% Causable rules for compound tasks")
        rules.append("")
        
        for task_name, methods in self.method_groups.items():
            for method in methods:
                task_atom = self._fmt_atom(method['task'])
                method_head = self._fmt_method_head(method, "T")
                subtasks = method['subtasks']
                preconditions = method['preconditions']

                # Get typing constraints for the compound task itself
                compound_task_typing = self._get_task_typing_constraints(method['task'])

                # Build precondition clauses
                precond_clauses = []
                for precond in preconditions:
                    if isinstance(precond, tuple) and precond[0] == 'not':
                        precond_clauses.append(f"not in_state({self._fmt_atom(precond[1])}, T)")
                    else:
                        precond_clauses.append(f"in_state({self._fmt_atom(precond)}, T)")

                # Get typing constraints
                precond_vars = self._get_variables_from_conditions(preconditions)
                typing_clauses = self._get_typing_constraints(precond_vars, method['params'])

                # Build causable constraints for all subtasks
                causable_clauses = []
                for i, subtask in enumerate(subtasks):
                    subtask_atom = self._fmt_atom(subtask)
                    if i == 0:
                        causable_clauses.append(f"causable({subtask_atom}, T, T2)")
                        causable_clauses.append("T2 >= T")
                    else:
                        causable_clauses.append(f"causable({subtask_atom}, T{i+1}, T{i+2})")
                        causable_clauses.append(f"T{i+2} >= T{i+1}")

                # Combine all body parts (with preconditions)
                body_parts = [f"{method_head}"] + precond_clauses + typing_clauses + compound_task_typing + causable_clauses
                body = ", ".join(body_parts)
                
                # Final time variable
                final_time = f"T{len(subtasks)+1}" if len(subtasks) != 0 else "T"
                
                causable_rule = f"causable({task_atom}, T, {final_time}) :- {body}."
                rules.append(causable_rule)
            
            rules.append("")
        
        return "\n".join(rules)

    def translate_problem(self):
        """Translate problem to ASP."""
        rules = []
        
        # Header comment
        rules.append("% Problem-specific facts")
        rules.append("")

        # Translate objects
        rules.append("% Object declarations")
        for obj, typ in self.data.objects.items():
            if typ != 'object':  # Don't add default type
                rules.append(f"{typ}({obj}).")
        rules.append("")

        # Translate initial state
        rules.append("% Initial state")
        for atom in self.data.initial_state:
            formatted_atom = self._fmt_atom(atom)
            rules.append(f"in_state({formatted_atom}, 0).")
        rules.append("")
        
        # Translate HTN tasks
        rules.append("% HTN Tasks")
        previous_task_atom = None
        previous_time_end = None
        
        for i, task in enumerate(self.data.htn_tasks):
            task_atom = self._fmt_atom(task)
            
            if i == 0:
                # First task starts at 0
                rules.append(f"taskTBA({task_atom}, 0).")
                previous_time_end = "T1" # Assuming first task ends at T1
            else:
                # Subsequent tasks depend on previous task
                # taskTBA(hi, Ti) :- causable(hi-1, Ti-1, Ti), Ti >= Ti-1.
                
                prev_task_atom = self._fmt_atom(self.data.htn_tasks[i-1])
                
                # So for i=1 (second task):
                # taskTBA(h1, T1) :- causable(h0, 0, T1), T1 >= 0.
                # Note: h0 starts at 0.
                
                # For i=2 (third task):
                # taskTBA(h2, T2) :- causable(h1, T1, T2), T2 >= T1.
                
                if i == 1:
                    rule = f"taskTBA({task_atom}, T1) :- causable({prev_task_atom}, 0, T1), T1 >= 0, time(T1)."
                else:
                    prev_start_var = f"T{i-1}"
                    current_start_var = f"T{i}"
                    rule = f"taskTBA({task_atom}, {current_start_var}) :- causable({prev_task_atom}, {prev_start_var}, {current_start_var}), {current_start_var} >= {prev_start_var}, time({prev_start_var}), time({current_start_var})."
                
                rules.append(rule)

        rules.append("")

        # Goal verification
        rules.append("% Goal verification")
        if self.data.htn_tasks:
            last_task = self.data.htn_tasks[-1]
            last_task_atom = self._fmt_atom(last_task)
            
            # plan_found :- causable(last_task, T_last_start, T_last_end), T_last_end >= T_last_start, <goal_conditions>.
            
            n_tasks = len(self.data.htn_tasks)
            if n_tasks == 1:
                start_var = "0"
                end_var = "T1"
            else:
                start_var = f"T{n_tasks-1}"
                end_var = f"T{n_tasks}"
            
            body_parts = [f"causable({last_task_atom}, {start_var}, {end_var})", f"{end_var} >= {start_var}", f"time({end_var})"]
            if start_var != "0":
                 body_parts.append(f"time({start_var})")

            # Add goal conditions
            for cond in self.data.goal_state:
                if isinstance(cond, tuple) and cond[0] == 'not':
                    body_parts.append(f"not in_state({self._fmt_atom(cond[1])}, {end_var})")
                else:
                    body_parts.append(f"in_state({self._fmt_atom(cond)}, {end_var})")
            
            rules.append(f"plan_found :- {', '.join(body_parts)}.")

        return "\n".join(rules)

    def export_primitives_list(self):
        """Export list of primitive task names."""
        return sorted(self.data.actions.keys())


def main():
    """Main entry point."""
    if len(sys.argv) == 5:
        domain_out_file = sys.argv[3]
        problem_out_file = sys.argv[4] 
    elif len(sys.argv) == 3:
        domain_out_file = "domain_output.lp"
        problem_out_file = "problem_output.lp"
    else:
        print("Usage: python hddl_parser.py <domain.hddl> <problem.hddl> <domain_output.lp> <problem_output.lp>")
        sys.exit(1)

    # Determine output directory for primitives.txt
    output_dir = os.path.dirname(domain_out_file) if os.path.dirname(domain_out_file) else "."
    primitives_out_file = os.path.join(output_dir, "primitives.txt")

    domain_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    
    
    # Parse HDDL
    parser = HDDLParser(domain_file, problem_file)
    parser.parse()
    
    # Translate to ASP
    translator = ASPTranslator(parser)
    
    # Translate and write domain
    domain_code = translator.translate_domain()
    with open(domain_out_file, 'w') as f:
        f.write(domain_code)
        
    # Translate and write problem
    problem_code = translator.translate_problem()
    with open(problem_out_file, 'w') as f:
        f.write(problem_code)

    # Export primitives list
    primitives = translator.export_primitives_list()
    with open(primitives_out_file, 'w') as f:
        for primitive in primitives:
            f.write(f"{primitive}\n")

    print(f"ASP domain translation written to: {domain_out_file}")
    print(f"ASP problem translation written to: {problem_out_file}")
    print(f"Primitives list written to: {primitives_out_file}")


if __name__ == '__main__':
    main()