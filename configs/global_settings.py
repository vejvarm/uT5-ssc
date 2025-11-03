import pathlib
import re

LOG_FOLDER = pathlib.Path("./logs/")
LOG_FOLDER.mkdir(exist_ok=True, parents=True)

CUSTOM_DTYPE_PATTERN = re.compile(r"(['\"])(.+?)\1\^\^(([\w\d_]+:)|(<[^>]+>))")