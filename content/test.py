#!/usr/bin/env python3
import sys

def strip_command_with_braces(text: str, command: str) -> str:
    """
    Removes all instances of \\<command>{...} and preserves the inner ...,
    correctly handling nested braces inside the argument.
    """
    token = "\\" + command + "{"
    out = []
    i = 0
    n = len(text)

    while i < n:
        if text.startswith(token, i):
            i += len(token)  # position right after the opening '{'
            depth = 1
            start = i

            while i < n and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                inner = text[start:i-1]  # exclude matching closing brace
                out.append(inner)
            else:
                # Unbalanced braces; keep the rest as-is to avoid corruption
                out.append(token + text[start:])
                break
        else:
            out.append(text[i])
            i += 1

    return "".join(out)

def main():
    data = sys.stdin.read()
    data = strip_command_with_braces(data, "mathbf")
    sys.stdout.write(data)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import sys

def strip_command_with_braces(text: str, command: str) -> str:
    """
    Removes all instances of \\<command>{...} and preserves the inner ...,
    correctly handling nested braces inside the argument.
    """
    token = "\\" + command + "{"
    out = []
    i = 0
    n = len(text)

    while i < n:
        if text.startswith(token, i):
            i += len(token)  # position right after the opening '{'
            depth = 1
            start = i

            while i < n and depth > 0:
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                i += 1

            if depth == 0:
                inner = text[start:i-1]  # exclude matching closing brace
                out.append(inner)
            else:
                # Unbalanced braces; keep the rest as-is to avoid corruption
                out.append(token + text[start:])
                break
        else:
            out.append(text[i])
            i += 1

    return "".join(out)

def main():
    data = sys.stdin.read()
    data = strip_command_with_braces(data, "mathbf")
    sys.stdout.write(data)

if __name__ == "__main__":
    main()

