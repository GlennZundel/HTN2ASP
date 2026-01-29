import sys
import re
import os
from collections import defaultdict


class HDDLParseError(Exception):
    """Exception for unsupported HDDL constructs."""
    pass


class HDDLParser:
    """Robust HDDL parser using balanced parenthesis matching."""

    KNOWN_DOMAIN_BLOCKS = {
        'define', 'domain', 'requirements', 'types', 'constants',
        'predicates', 'task', 'action', 'method'
    }
    KNOWN_PROBLEM_BLOCKS = {
        'define', 'problem', 'domain', 'objects', 'init', 'htn', 'goal'
    }
    KNOWN_CONDITION_KEYWORDS = {'and', 'not'}
    UNSUPPORTED_CONDITION_KEYWORDS = {'or', 'forall', 'exists', 'imply', 'when', 'for'}

    def __init__(self, domain_file, problem_file):
        self.domain_content = self._read_file(domain_file)
        self.problem_content = self._read_file(problem_file)
        
        self.types = {}
        self.predicates = {}
        self.tasks = {}
        self.actions = {}
        self.methods = []
        self.constants = {}  # Constants aus Domain

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

        # Check for unsupported condition keywords
        first_word = sexp_str.split()[0].lower() if sexp_str.split() else ''
        if first_word in self.UNSUPPORTED_CONDITION_KEYWORDS:
            raise HDDLParseError(
                f"Nicht unterstütztes HDDL-Konstrukt: '{first_word}'. "
                f"Nur 'and' und 'not' werden unterstützt."
            )

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
                    # Normalize "not(" to "not (" for consistent parsing
                    if inner.lower().startswith('not('):
                        inner = 'not (' + inner[4:]
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

    def _validate_domain_blocks(self, content):
        """Check for unknown top-level blocks in domain."""
        for match in re.finditer(r'\(:(\w+)', content):
            keyword = match.group(1).lower()
            if keyword not in self.KNOWN_DOMAIN_BLOCKS:
                raise HDDLParseError(
                    f"Unbekannter HDDL-Block in Domain: ':{keyword}'. "
                    f"Unterstützte Blöcke: {', '.join(':' + k for k in sorted(self.KNOWN_DOMAIN_BLOCKS))}"
                )

    def _validate_problem_blocks(self, content):
        """Check for unknown top-level blocks in problem."""
        for match in re.finditer(r'\(:(\w+)', content):
            keyword = match.group(1).lower()
            if keyword not in self.KNOWN_PROBLEM_BLOCKS:
                raise HDDLParseError(
                    f"Unbekannter HDDL-Block in Problem: ':{keyword}'. "
                    f"Unterstützte Blöcke: {', '.join(':' + k for k in sorted(self.KNOWN_PROBLEM_BLOCKS))}"
                )

    def parse(self):
        """Main parsing."""
        self._parse_domain()
        self._parse_problem()

    def _parse_domain(self):
        """Parse domain file."""
        content = self.domain_content

        # Validate domain blocks
        self._validate_domain_blocks(content)

        # Parse types
        types_start_match = re.search(r'\(:types\s', content, re.IGNORECASE)
        if types_start_match:
             start = types_start_match.start()
             block, _ = self._extract_balanced(content, start)
             if block:
                 # Remove '(:types' and closing ')'
                 # We need to be careful about potential whitespace after :types
                 keyword_len = len('(:types')
                 # Find where the keyword ends exactly just in case of whitespace
                 inner_start = block.lower().find('(:types') + keyword_len
                 inner = block[inner_start:-1].strip()
                 typed_list = self._parse_typed_list(inner)
                 for item, parent in typed_list:
                     self.types[item] = parent

        # Parse constants
        constants_match = re.search(r'\(:constants\s', content, re.IGNORECASE)
        if constants_match:
            start = constants_match.start()
            block, _ = self._extract_balanced(content, start)
            if block:
                keyword_len = len('(:constants')
                inner_start = block.lower().find('(:constants') + keyword_len
                inner = block[inner_start:-1].strip()
                typed_list = self._parse_typed_list(inner)
                for const, typ in typed_list:
                    self.constants[const] = typ

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
        pos = 0
        while True:
            match = re.search(r'\(:task\s', content[pos:], re.IGNORECASE)
            if not match:
                break
            
            task_start = pos + match.start()
            
            task_block, next_pos = self._extract_balanced(content, task_start)
            if task_block:
                tokens = self._tokenize_sexp(task_block)
                if len(tokens) >= 2:
                    # tokens[0] is ':task'
                    task_name = tokens[1].lower()
                    
                    params = []
                    # Find :parameters in tokens
                    for i in range(2, len(tokens)):
                        if tokens[i].lower() == ':parameters':
                            if i + 1 < len(tokens):
                                params_expr = tokens[i+1]
                                # check if it is a parenthesized list
                                if params_expr.startswith('(') and params_expr.endswith(')'):
                                     params_text = params_expr[1:-1].strip()
                                     params = self._parse_typed_list(params_text)
                            break
                    
                    self.tasks[task_name] = params
            
            pos = task_start + len(task_block) if task_block else pos + 1
        
        # Parse actions
        pos = 0
        while True:
            match = re.search(r'\(:action\s+([^\s]+)', content[pos:], re.IGNORECASE)
            if not match:
                break
            
            action_start = pos + match.start()
            action_name = match.group(1).lower().replace('-', '_')

            
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
                subtasks_match = re.search(r':(?:ordered-tasks|ordered-subtasks|subtasks|tasks)\s*\(', method_block, re.IGNORECASE)
                if subtasks_match:
                    block_lower = method_block.lower()
                    if ':ordered-tasks' in block_lower:
                        keyword = ':ordered-tasks'
                    elif ':ordered-subtasks' in block_lower:
                        keyword = ':ordered-subtasks'
                    elif ':subtasks' in block_lower:
                        keyword = ':subtasks'
                    else:
                        keyword = ':tasks'
                    subtasks_start = block_lower.find(keyword, 0, len(method_block))
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
                                            tokens = self._tokenize_sexp(st_expr)
                                            if len(tokens) == 2 and tokens[1].startswith('('):
                                                # Labeled task: (label (task ...))
                                                subtasks.append(self._parse_atom(tokens[1]))
                                            else:
                                                # Unlabeled task
                                                subtasks.append(tuple(t.lower() for t in tokens))
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

            # Validate problem blocks
            self._validate_problem_blocks(content)

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
        self.equality_types = set()  # Types used in equality predicates (=)

        # Rename constant 't' to 'tt' to avoid collision with time parameter
        self.constant_renames = {}
        if 't' in self.data.constants:
            self.constant_renames['t'] = 'tt'

        # Determine unique variable names to avoid collisions with domain variables
        domain_vars = self._get_all_domain_vars()
        self.time_var = self._determine_unique_var("T", domain_vars)
        self.gen_var = self._determine_unique_var("X", domain_vars)

    def _get_all_domain_vars(self):
        """Collect all variable names used in the domain/problem."""
        vars = set()
        for action in self.data.actions.values():
            for p, _ in action['params']: vars.add(self._fmt_term(p))
        for params in self.data.tasks.values():
            for p, _ in params: vars.add(self._fmt_term(p))
        for method in self.data.methods:
            for p, _ in method['params']: vars.add(self._fmt_term(p))
        for t in self.data.types: vars.add(self._fmt_term(t))
        return vars

    def _determine_unique_var(self, base, forbidden, max_suffix=20):
        """Find unique variable name that doesn't collide with domain vars.

        Checks base and numbered variants (base2, base3, ..., base{max_suffix})
        to ensure no collisions with forbidden names.
        """
        candidate = base
        while True:
            # Check base and all numbered variants
            variants = [candidate] + [f"{candidate}{i}" for i in range(2, max_suffix + 1)]
            if not any(v in forbidden for v in variants):
                return candidate
            # Try next candidate by appending base
            candidate += base

    def _collect_equality_types(self):
        """Collect all types used in equality predicates (=)."""
        def process_conditions(conditions, param_list):
            type_map = {p[0]: p[1] for p in param_list}
            for cond in conditions:
                atom = cond[1] if isinstance(cond, tuple) and cond[0] == 'not' else cond
                if atom and atom[0] == '=':
                    # Equality predicate found
                    for arg in atom[1:]:
                        if arg.startswith('?') and arg in type_map:
                            typ = type_map[arg]
                            if typ != 'object':
                                self.equality_types.add(typ)

        # Search through actions
        for action in self.data.actions.values():
            process_conditions(action['pre'], action['params'])
            process_conditions(action['eff'], action['params'])

        # Search through methods
        for method in self.data.methods:
            process_conditions(method['preconditions'], method['params'])

    def _preprocess_actions_with_preconditions(self):
        """
        Für jede action mit preconditions: erstelle compound task wrapper + methode.
        Returns: dict mapping original action name to wrapper task name
        """
        action_wrappers = {}
        new_methods = []
        
        for action_name, action in self.data.actions.items():
            if action['pre']:  # Hat preconditions
                action_name = action_name.lower().replace('-', '_')
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
                task_name = subtask[0].lower().replace('-', '_')
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
        # Rename constants that collide with time parameter
        term_lower = term.lower().replace('-', '_')
        if term_lower in self.constant_renames:
            return self.constant_renames[term_lower]
        return term_lower

    def _fmt_atom(self, atom):
        """Format atom as ASP."""
        if not atom:
            return ""
        name = atom[0]
        # Special case: = becomes equals
        if name == '=':
            name = 'equals'
        else:
            name = name.replace('-', '_')  # Replace hyphens with underscores
        params = [self._fmt_term(p) for p in atom[1:]]
        if not params:
            return name
        return f"{name}({', '.join(params)})"

    def _fmt_atom_with_id(self, atom, id_expr):
        """Format atom with ID as last parameter."""
        base_atom = self._fmt_atom(atom)
        if '(' in base_atom and base_atom.endswith(')'):
            return f"{base_atom[:-1]}, {id_expr})"
        else:
            # Atom without parameters
            return f"{base_atom}({id_expr})"

    def _build_id_call(self, method_name, task_params, method_params, index, time_var):
        """Build the @id(...) call for unique subtask identification."""
        all_args = [f'"{method_name}"']
        all_args.extend(task_params)
        all_args.extend(method_params)
        all_args.append(str(index))
        all_args.append(time_var)
        return f"@id({', '.join(all_args)})"

    def _deduplicate_body_parts(self, body_parts):
        """Remove duplicate entries while preserving order."""
        seen = set()
        result = []
        for part in body_parts:
            if part not in seen:
                seen.add(part)
                result.append(part)
        return result

    def _get_time_var(self, n):
        """Generate time variable name using unique base."""
        if n <= 1:
            return self.time_var
        return f"{self.time_var}{n}"

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
                    typing_clauses.append(f"{typ.replace('-', '_')}({self._fmt_term(var)})")

        return typing_clauses

    def _get_typing_from_predicate(self, atom, variables):
        """Generate typing constraints for variables based on predicate definition.

        Args:
            atom: Tuple like ('at', '?c', '?p')
            variables: Set of variable names to get typing for (e.g., {'?p'})

        Returns:
            List of typing constraints like ['place(P)']
        """
        if not atom or len(atom) < 2:
            return []

        pred_name = atom[0]
        pred_args = atom[1:]

        # Look up predicate definition
        if pred_name not in self.data.predicates:
            return []

        pred_params = self.data.predicates[pred_name]

        # Generate typing for requested variables
        typing_clauses = []
        for i, arg in enumerate(pred_args):
            if arg in variables and i < len(pred_params):
                param_name, param_type = pred_params[i]
                if param_type != 'object':
                    typing_clauses.append(f"{param_type.replace('-', '_')}({self._fmt_term(arg)})")

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
                    typing_clauses.append(f"{typ.replace('-', '_')}({self._fmt_term(param)})")
        
        return typing_clauses

    def _build_check_clauses(self, method, hasSubtasks):
        """Build checked_state clauses for a method.

        For methods with preconditions: returns clauses referencing checked_state for each precondition.
        For methods without preconditions: returns a clause referencing a synthetic typing checked_state.
        """
        method_term_name = method['name'].replace('-', '_')
        check_clauses = []

        if method['preconditions']:
            # Normal checked_state for each precondition
            for idx, precond in enumerate(method['preconditions']):
                check_pred = f"checked_state_{method_term_name}_{idx}"
                if isinstance(precond, tuple) and precond[0] == 'not':
                    raw_atom = precond[1]
                else:
                    raw_atom = precond
                fmt_atom = self._fmt_atom(raw_atom)
                temp_var = f"{check_pred}({fmt_atom}, {self.time_var})" if hasSubtasks else f"{check_pred}({fmt_atom}, t)"
                check_clauses.append(temp_var)
        else:
            # Synthetic checked_state for typing - use ALL method params
            check_pred = f"checked_state_{method_term_name}_typing"
            fmt_vars = [self._fmt_term(p[0]) for p in method['params']]
            head_args = fmt_vars + [self.time_var] if hasSubtasks else fmt_vars + ["t"]
            check_clauses.append(f"{check_pred}({', '.join(head_args)})")

        return check_clauses

    def _fmt_method_head(self, method, time_var, id_var="ID"):
        """Format method head with compound task atom (including ID), method-only params, and time."""
        name = method['name'].replace('-', '_')

        # Format the full compound task atom with ID (e.g., "do_put_on(C, S2, ID)" or just "nop(ID)")
        task_atom_with_id = ""
        if method['task']:
            task_atom_with_id = self._fmt_atom_with_id(method['task'], id_var)

        # Get task variables to exclude from method-only params
        task_vars = set()
        if method['task']:
            for term in method['task'][1:]:
                if term.startswith('?'):
                    task_vars.add(term)

        # Get method-only params (not in task)
        method_only_params = []
        for param_name, _ in method['params']:
            if param_name not in task_vars:
                method_only_params.append(self._fmt_term(param_name))

        parts = []
        if task_atom_with_id:
            parts.append(task_atom_with_id)
        parts.extend(method_only_params)
        parts.append(str(time_var))

        return f"{name}({', '.join(parts)})"

    def _get_method_typing_constraints(self, method):
        """Generate typing constraints for method parameters."""
        constraints = []
        for param_name, param_type in method['params']:
            if param_type != 'object':
                constraints.append(f"{param_type.replace('-', '_')}({self._fmt_term(param_name)})")
        return constraints

    def _get_method_condition(self, method):
        """Get typing constraints for method (task params only) for choice rules."""
        constraints = []

        # Task parameter typing
        if method['task']:
            constraints.extend(self._get_task_typing_constraints(method['task']))

        return self._deduplicate_body_parts(constraints)

    def translate_domain(self):
        """Translate domain to ASP."""
        self._collect_equality_types()
        action_wrappers = self._preprocess_actions_with_preconditions()
        self._replace_action_references_in_methods(action_wrappers)

        rules = []

        # Python script for unique ID generation
        rules.append("#script (python)")
        rules.append("import hashlib")
        rules.append("import clingo")
        rules.append("")
        rules.append("def id(*args):")
        rules.append("    input_str = '#'.join(str(arg) for arg in args)")
        rules.append("    hash_val = hashlib.sha256(input_str.encode()).hexdigest()[:12]")
        rules.append("    return clingo.String(hash_val)")
        rules.append("#end.")
        rules.append("")

        # Header comment
        rules.append("#program base.")
        rules.append("% Primitive tasks (actions)")
        rules.append("")

        # Type hierarchy
        rules.append("% Type hierarchy")
        for subtype, supertype in self.data.types.items():
            if supertype != 'object':
                # Format atoms to handle hyphens
                subtype_fmt = subtype.replace('-', '_')
                supertype_fmt = supertype.replace('-', '_')
                rules.append(f"{supertype_fmt}({self.gen_var}) :- {subtype_fmt}({self.gen_var}).")
        rules.append("")

        # Constant declarations (from domain)
        if self.data.constants:
            rules.append("% Constant declarations")
            for const, typ in self.data.constants.items():
                const_fmt = const.replace('-', '_')
                # Rename 't' to 'tt' to avoid collision with time parameter
                if const_fmt in self.constant_renames:
                    const_fmt = self.constant_renames[const_fmt]
                if typ != 'object':
                    rules.append(f"{typ.replace('-', '_')}({const_fmt}).")
                else:
                    # Für Konstanten ohne expliziten Typ nur als Fakt ausgeben
                    rules.append(f"constant({const_fmt}).")
            rules.append("")

        # Equality rules for types used in = predicates
        if self.equality_types:
            rules.append("% Equality rules")
            for typ in sorted(self.equality_types):
                typ_fmt = typ.replace('-', '_')
                rules.append(f"in_state(equals(A, A), 0) :-{typ_fmt}(A).")
            rules.append("")

        # Atom definitions for predicates
        rules.append("% Atom definitions")
        for pred_name, params in self.data.predicates.items():
            # Build predicate atom
            if params:
                # With parameters: atom(pred(X, Y)) :- type1(X), type2(Y).
                param_vars = [self._fmt_term(p[0]) for p in params]
                atom_str = f"{pred_name.replace('-', '_')}({', '.join(param_vars)})"

                # Build typing constraints
                typing_constraints = []
                for param_name, param_type in params:
                    if param_type != 'object':
                        typing_constraints.append(f"{param_type.replace('-', '_')}({self._fmt_term(param_name)})")

                if typing_constraints:
                    rules.append(f"atom({atom_str}) :- {', '.join(typing_constraints)}.")
                else:
                    rules.append(f"atom({atom_str}).")
            else:
                # No parameters: atom(pred).
                rules.append(f"atom({pred_name.replace('-', '_')}).")
        rules.append("")

        #here starts the program step part that gets incremented
        rules.append("#program step(t).")

        # Translate primitive tasks (actions)
        for action_name, action in self.data.actions.items():
            # Build action atom with ID parameter
            param_names = tuple(p[0] for p in action['params'])
            action_atom_tuple = (action_name,) + param_names
            action_atom = self._fmt_atom(action_atom_tuple)
            action_atom_with_id = self._fmt_atom_with_id(action_atom_tuple, "ID")

            # Get typing constraints for action parameters
            action_typing = self._get_task_typing_constraints(action_atom_tuple)

            # Build body for causable rule - ID comes from taskTBA
            body_parts = [f"taskTBA({action_atom_with_id}, t-1)"] + action_typing
            body = ", ".join(body_parts)

            # causable rule for primitive task (with ID)
            causable_rule = f"causable({action_atom_with_id}, t-1, t) :- {body}."
            rules.append(causable_rule)

            # Effects (also use action_atom_with_id for taskTBA reference)
            for eff in action['eff']:
                if isinstance(eff, tuple) and eff[0] == 'not':
                    # Delete effect
                    eff_atom = self._fmt_atom(eff[1])
                    delete_rule = f"out_state({eff_atom}, t) :- taskTBA({action_atom_with_id}, t-1)."
                    rules.append(delete_rule)
                else:
                    # Add effect
                    eff_atom = self._fmt_atom(eff)
                    add_rule = f"in_state({eff_atom}, t) :- taskTBA({action_atom_with_id}, t-1)."
                    rules.append(add_rule)

            rules.append("")
        
        # Translate methods
        rules.append("% Method selection rules")
        rules.append("")

        for task_name, methods in self.method_groups.items():
            # Get task atom with ID from first method (all methods decompose the same task)
            task_atom_with_id = self._fmt_atom_with_id(methods[0]['task'], "ID")

            if len(methods) == 1:
                method = methods[0]
                method_head = self._fmt_method_head(method, "t")
                typing = self._get_method_typing_constraints(method)

                # Check if method has own params (not in task)
                task_vars = set()
                if method['task']:
                    for term in method['task'][1:]:
                        if term.startswith('?'):
                            task_vars.add(term)
                has_own_params = any(p[0] not in task_vars for p in method['params'])

                if has_own_params:
                    # Choice rule needed to bind extra params
                    if typing:
                        alt = f"{method_head} : {', '.join(typing)}"
                    else:
                        alt = method_head
                    rule = f"1 {{ {alt} }} 1 :- time(t), taskTBA({task_atom_with_id}, t)."
                else:
                    # Simple rule - all params bound by taskTBA
                    body_parts = [f"time(t)", f"taskTBA({task_atom_with_id}, t)"] + typing
                    rule = f"{method_head} :- {', '.join(body_parts)}."
                rules.append(rule)
            else:
                # Multiple methods - build choice rule
                alternatives = []
                for method in methods:
                    method_head = self._fmt_method_head(method, "t")
                    conditions = self._get_method_typing_constraints(method)

                    if conditions:
                        alt = f"{method_head} : {', '.join(conditions)}"
                    else:
                        alt = method_head
                    alternatives.append(alt)

                choice_body = "; ".join(alternatives)
                rule = f"1 {{ {choice_body} }} 1 :- time(t), taskTBA({task_atom_with_id}, t)."
                rules.append(rule)

            rules.append("")
        
        # Generate Checked State Rules (Definition 27)
        rules.append("% Checked State rules")
        rules.append("")
        
        for task_name, methods in self.method_groups.items():
            for method in methods:
                method_head = self._fmt_method_head(method, "t")
                method_term_name = method['name'].replace('-', '_')

                # Get variables from compound task head (these are "bound")
                compound_task_vars = set()
                if method['task']:
                    for term in method['task'][1:]:  # Skip task name
                        if term.startswith('?'):
                            compound_task_vars.add(term)

                # Get variables from method parameters (also "bound")
                method_param_vars = {p[0] for p in method['params']}

                # All bound variables = task params + method params
                bound_vars = compound_task_vars.union(method_param_vars)

                for idx, precond in enumerate(method['preconditions']):
                    check_pred = f"checked_state_{method_term_name}_{idx}"

                    # Extract atom for the head
                    if isinstance(precond, tuple) and precond[0] == 'not':
                        raw_atom = precond[1]
                        is_negated = True
                    else:
                        raw_atom = precond
                        is_negated = False

                    fmt_atom = self._fmt_atom(raw_atom)

                    # Head: checked_state_...(Atom, T)
                    check_atom_head = f"{check_pred}({fmt_atom}, t)"

                    # Get variables from precondition
                    raw_vars = self._get_variables_from_conditions([precond])

                    # Identify unbound variables (not in task head or method params)
                    unbound_vars = raw_vars - bound_vars

                    if not unbound_vars:
                        # Case 1: All variables bound - normal rule
                        check_body = [method_head]
                        if is_negated:
                            check_body.append(f"not in_state({fmt_atom}, t)")
                        else:
                            check_body.append(f"in_state({fmt_atom}, t)")
                        rules.append(f"{check_atom_head} :- {', '.join(check_body)}.")
                    else:
                        # Case 2: Unbound variables - choice rule
                        # Get typing constraints for unbound variables from predicate definition
                        typing_constraints = self._get_typing_from_predicate(raw_atom, unbound_vars)

                        # in_state or not in_state
                        if is_negated:
                            state_check = f"not in_state({fmt_atom}, t)"
                        else:
                            state_check = f"in_state({fmt_atom}, t)"

                        # Build choice rule: 1 { head : typing, state_check } 1 :- method_head.
                        condition_parts = typing_constraints + [state_check]
                        choice_body = f"{check_atom_head} : {', '.join(condition_parts)}"
                        rules.append(f"1 {{ {choice_body} }} 1 :- {method_head}.")

                # Synthetic checked_state for methods without preconditions
                if not method['preconditions']:
                    check_pred = f"checked_state_{method_term_name}_typing"

                    # Use ALL method params for synthetic checked_state
                    fmt_vars = [self._fmt_term(p[0]) for p in method['params']]

                    # Head: checked_state_..._typing(X, Y, ..., T)
                    head_args = fmt_vars + ["t"]
                    check_atom_head = f"{check_pred}({', '.join(head_args)})"

                    # Synthetic checked_state always has all variables bound (they're all method params in method_head)
                    # Therefore, always use a simple rule, not a choice rule
                    rules.append(f"{check_atom_head} :- {method_head}.")
        rules.append("")

        # Translate subtasks for each method
        rules.append("% Subtask rules")
        rules.append("")

        for task_name, methods in self.method_groups.items():
            for method in methods:
                method_head = self._fmt_method_head(method, self.time_var)
                subtasks = method['subtasks']

                # Build checking clauses (using helper function)
                check_clauses = self._build_check_clauses(method, True)

                # Extract task parameter variables for @id
                task_param_vars = [self._fmt_term(p) for p in method['task'][1:] if p.startswith('?')]
                # Extract method-only parameter variables for @id
                task_vars_set = set(method['task'][1:]) if method['task'] else set()
                method_only_param_vars = [self._fmt_term(p[0]) for p in method['params'] if p[0] not in task_vars_set]

                # Subtask rules generation
                for i, subtask in enumerate(subtasks):
                    # Build @id for this subtask
                    # For i=0: time is 't' (start of first subtask)
                    # For i>0: time is T (method start time)
                    id_time_var = "t" if i == 0 else self.time_var
                    subtask_id = self._build_id_call(
                        method['name'],
                        task_param_vars,
                        method_only_param_vars,
                        i,
                        id_time_var
                    )
                    subtask_atom_with_id = self._fmt_atom_with_id(subtask, subtask_id)

                    # Get typing constraints for the subtask's parameters
                    task_typing_clauses = self._get_task_typing_constraints(subtask)

                    # Build rule body: time + typing (early!) + method + checks
                    if i == 0:
                        body_parts = [f"time(t)"] + task_typing_clauses + [self._fmt_method_head(method, 't')] + self._build_check_clauses(method, False)

                    # Add causable constraint for immediately previous subtask
                    if i > 0:
                        body_parts = [f"time(t)"] + task_typing_clauses + [method_head] + check_clauses
                        # Build @id for previous subtask - ALWAYS use T (method start time)
                        prev_subtask_id = self._build_id_call(
                            method['name'],
                            task_param_vars,
                            method_only_param_vars,
                            i - 1,
                            self.time_var  # Always T for prev subtask
                        )
                        prev_subtask_with_id = self._fmt_atom_with_id(subtasks[i-1], prev_subtask_id)
                        prev_time = self._get_time_var(i)
                        curr_time = "t"
                        body_parts.append(f"causable({prev_subtask_with_id}, {prev_time}, {curr_time})")
                        body_parts.append(f"{curr_time} >= {prev_time}")

                    # Deduplicate to remove redundant typing constraints
                    body_parts = self._deduplicate_body_parts(body_parts)

                    body = ", ".join(body_parts)

                    # Regular rule for all subtasks (no choice rules)
                    subtask_rule = f"taskTBA({subtask_atom_with_id}, t) :- {body}."

                    rules.append(subtask_rule)

                rules.append("")
        
        # Causable rules for compound tasks
        rules.append("% Causable rules for compound tasks")
        rules.append("")

        for task_name, methods in self.method_groups.items():
            for method in methods:
                # Compound task atom with ID (ID is variable from method head)
                task_atom_with_id = self._fmt_atom_with_id(method['task'], "ID")
                subtasks = method['subtasks']
                method_head = self._fmt_method_head(method, self.time_var) if len(subtasks) != 0 else self._fmt_method_head(method, "t")

                # Build checking clauses (using helper function)
                check_clauses = self._build_check_clauses(method, True) if len(subtasks) != 0 else self._build_check_clauses(method, False)

                # Get typing constraints for the compound task itself
                compound_task_typing = self._get_task_typing_constraints(method['task'])

                # Get typing constraints for method params
                method_param_names = [p[0] for p in method['params']]
                typing_clauses = self._get_typing_constraints(method_param_names, method['params'])

                # Extract task parameter variables and method-only params for @id
                task_param_vars = [self._fmt_term(p) for p in method['task'][1:] if p.startswith('?')]
                task_vars_set = set(method['task'][1:]) if method['task'] else set()
                method_only_param_vars = [self._fmt_term(p[0]) for p in method['params'] if p[0] not in task_vars_set]

                # Build causable constraint for LAST subtask only
                causable_clauses = []
                if subtasks:
                    last_subtask = subtasks[-1]
                    # Build @id for last subtask - use T (method start time)
                    last_subtask_id = self._build_id_call(
                        method['name'],
                        task_param_vars,
                        method_only_param_vars,
                        len(subtasks) - 1,
                        self.time_var  # T
                    )
                    last_subtask_with_id = self._fmt_atom_with_id(last_subtask, last_subtask_id)

                    # Calculate time variables for last subtask
                    n = len(subtasks)
                    if n == 1:
                        start_var = self.time_var
                        end_var = "t"
                    else:
                        start_var = self._get_time_var(n)
                        end_var = "t"

                    causable_clauses.append(f"causable({last_subtask_with_id}, {start_var}, {end_var})")
                    causable_clauses.append(f"{end_var} >= {start_var}")

                # Combine all body parts: typing early for better grounding
                body_parts = typing_clauses + compound_task_typing + [method_head] + check_clauses + causable_clauses
                body = ", ".join(self._deduplicate_body_parts(body_parts))

                # Final time variable
                final_time = "t"
                start_time = self.time_var if len(subtasks) != 0 else "t"
                # Causable rule for compound task (with ID)
                causable_rule = f"causable({task_atom_with_id}, {start_time}, {final_time}) :- {body}."
                rules.append(causable_rule)

            rules.append("")

        return "\n".join(rules)

    def translate_problem(self):
        """Translate problem to ASP."""
        rules = []
        #object declarations and initial state definitions dont change over time
        rules.append("#program base.")
        
        # Translate objects
        rules.append("% Object declarations")
        for obj, typ in self.data.objects.items():
            if typ != 'object':  # Don't add default type
                rules.append(f"{typ.replace('-', '_')}({self._fmt_term(obj)}).")
        rules.append("")

        # Translate initial state
        rules.append("% Initial state")
        for atom in self.data.initial_state:
            formatted_atom = self._fmt_atom(atom)
            rules.append(f"in_state({formatted_atom}, 0).")
        rules.append("")

        #this part changes with time
        # Translate HTN tasks
        rules.append("% HTN Tasks")

        for i, task in enumerate(self.data.htn_tasks):
            # Use "init_X" string ID for initial tasks
            init_id = f'"init_{i}"'
            task_atom_with_id = self._fmt_atom_with_id(task, init_id)

            if i == 0:
                # First task starts at time 0
                rules.append(f"taskTBA({task_atom_with_id}, 0).")
                rules.append("#program step(t).")
            else:
                # Subsequent tasks depend on previous task
                prev_init_id = f'"init_{i-1}"'
                prev_task_with_id = self._fmt_atom_with_id(self.data.htn_tasks[i-1], prev_init_id)

                if i == 1:
                    start_var = "0"
                    end_var = "t"
                    causable_call = f"causable({prev_task_with_id}, {start_var}, {end_var})"
                    rule = f"taskTBA({task_atom_with_id}, {end_var}) :- {causable_call}, {end_var} >= {start_var}, time({end_var})."
                else:
                    prev_start_var = self._get_time_var(i-1)
                    current_start_var = "t"
                    causable_call = f"causable({prev_task_with_id}, {prev_start_var}, {current_start_var})"
                    rule = f"taskTBA({task_atom_with_id}, {current_start_var}) :- {causable_call}, {current_start_var} >= {prev_start_var}, time({prev_start_var}), time({current_start_var})."

                rules.append(rule)

        rules.append("")

        # Goal verification
        rules.append("% Goal verification")
        if self.data.htn_tasks:
            last_task = self.data.htn_tasks[-1]
            # Use "init_X" ID for last task
            last_init_id = f'"init_{len(self.data.htn_tasks)-1}"'
            last_task_with_id = self._fmt_atom_with_id(last_task, last_init_id)

            n_tasks = len(self.data.htn_tasks)
            if n_tasks == 1:
                start_var = "0"
                end_var = "t"
            else:
                start_var = self._get_time_var(n_tasks-1)
                end_var = "t"

            # Causable call for the last task (with init ID)
            causable_call = f"causable({last_task_with_id}, {start_var}, {end_var})"

            body_parts = [causable_call, f"{end_var} >= {start_var}", f"time({end_var})"]
            if start_var != "0":
                body_parts.append(f"time({start_var})")

            # Add goal conditions
            for cond in self.data.goal_state:
                if isinstance(cond, tuple) and cond[0] == 'not':
                    body_parts.append(f"not in_state({self._fmt_atom(cond[1])}, {end_var})")
                else:
                    body_parts.append(f"in_state({self._fmt_atom(cond)}, {end_var})")
            
            rules.append(f"plan_found(t) :- {', '.join(body_parts)}.")

        return "\n".join(rules)

    def export_primitives_list(self):
        """Export list of primitive task names."""
        return sorted(self.data.actions.keys())


def main():
    """Main entry point."""
    # Example usage:
    # python3 hddl_to_lp.py domain.hddl problem.hddl domain_output.lp problem_output.lp primitives.txt

    if len(sys.argv) != 6:
        print("Usage: hddl_to_lp.py <domain> <problem> <domain_output> <problem_output> <primitives_output>")
        print("  <domain>            - Input HDDL domain file")
        print("  <problem>           - Input HDDL problem file")
        print("  <domain_output>     - Output ASP domain file")
        print("  <problem_output>    - Output ASP problem file")
        print("  <primitives_output> - Output file listing primitive task names")
        sys.exit(1)

    domain_file = sys.argv[1]
    problem_file = sys.argv[2]
    domain_out_file = sys.argv[3]
    problem_out_file = sys.argv[4]
    primitives_out_file = sys.argv[5]

    try:
        # Parse HDDL
        parser = HDDLParser(domain_file, problem_file)
        parser.parse()
    except HDDLParseError as e:
        print(f"FEHLER: {e}", file=sys.stderr)
        sys.exit(1)

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