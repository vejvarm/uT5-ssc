from datetime import datetime

from configs.global_settings import CUSTOM_DTYPE_PATTERN, LOG_FOLDER

def log(msg: str, log_file_name: str):
    log_msg = f"{datetime.now()}: {msg}\n"

    log_file = LOG_FOLDER.joinpath(log_file_name)
    if not log_file.exists():
        with log_file.open("w") as f:
            f.write(log_msg)
    else:
        with log_file.open("a") as f:
            f.write(log_msg)

def replace_custom_datatypes(sparql, keep_xsd=True):
    # If keep_xsd: replace with ^^xsd:string, else just strip datatype (keep as plain literal)
    sparql = sparql.replace("^^ ", "^^")
    if keep_xsd:
        def repl(m):
            return f"\"{m.group(2)}\"^^xsd:string"
    else:
        def repl(m):
            return f"\"{m.group(2)}\""
    return CUSTOM_DTYPE_PATTERN.sub(repl, sparql)