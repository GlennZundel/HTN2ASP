import re, sys

def process_tasks(names_file, data_file, result_file):

    with open(names_file, 'r', encoding='utf-8') as f:
        valid_names = {line.strip() for line in f if line.strip()}

    with open(data_file, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    tokens = raw_content.split()
    parsed_items = []
    regex_pattern = re.compile(r'taskTBA\((.+?),(\d+)\)')
    for token in tokens:
        match = regex_pattern.match(token)
        if match:
            action_body = match.group(1)
            number = int(match.group(2))
            func_name_match = re.match(r'^([a-zA-Z0-9_]+)(\(|$)', action_body)
            if func_name_match:
                func_name = func_name_match.group(1)
                if func_name in valid_names:
                    if '(' in action_body:
                        formatted_string = f"{action_body[:-1]}, {number})"
                    else:
                        formatted_string = f"{action_body}({number})"
                    parsed_items.append((number, formatted_string))
    parsed_items.sort(key=lambda x: x[0])
    with open(result_file, 'w', encoding='utf-8') as f:
        for _, line in parsed_items:
            f.write(line + '\n')
    print(f"{len(parsed_items)}  Actions gefunden")


if __name__ == "__main__":
    process_tasks(sys.argv[1], sys.argv[2], sys.argv[3])
