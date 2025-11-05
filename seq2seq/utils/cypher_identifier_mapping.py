import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_BOUNDARY = r"[A-Za-z0-9_]"
CONTEXT_LABEL = "label"
CONTEXT_PROPERTY = "property"


def _build_token_pattern(tokens: Iterable[str]) -> Optional[re.Pattern]:
    escaped = [re.escape(tok) for tok in tokens if tok]
    if not escaped:
        return None
    escaped.sort(key=len, reverse=True)
    joined = "|".join(escaped)
    return re.compile(rf"(?<!{TOKEN_BOUNDARY})(?:{joined})(?!{TOKEN_BOUNDARY})")


def _build_label_pattern(tokens: Iterable[str]) -> Optional[re.Pattern]:
    escaped = [re.escape(tok) for tok in tokens if tok]
    if not escaped:
        return None
    escaped.sort(key=len, reverse=True)
    joined = "|".join(escaped)
    return re.compile(rf"(?P<prefix>:\s*`?)(?P<token>{joined})(?P<suffix>`?)")


def _build_property_accessor_pattern(tokens: Iterable[str]) -> Optional[re.Pattern]:
    escaped = [re.escape(tok) for tok in tokens if tok]
    if not escaped:
        return None
    escaped.sort(key=len, reverse=True)
    joined = "|".join(escaped)
    return re.compile(rf"(?P<prefix>\.\s*`?)(?P<token>{joined})(?P<suffix>`?)")


def _build_property_map_pattern(tokens: Iterable[str]) -> Optional[re.Pattern]:
    escaped = [re.escape(tok) for tok in tokens if tok]
    if not escaped:
        return None
    escaped.sort(key=len, reverse=True)
    joined = "|".join(escaped)
    return re.compile(rf"(?P<prefix>[\{{,\s]\s*`?)(?P<token>{joined})(?P<suffix>`?\s*:)")


def _replace_in_segment(segment: str, mapping: Dict[str, str], pattern: Optional[re.Pattern]) -> str:
    if not segment or not mapping or pattern is None:
        return segment
    return pattern.sub(lambda match: mapping.get(match.group(0), match.group(0)), segment)


def _replace_outside_string_literals(text: str, mapping: Dict[str, str], pattern: Optional[re.Pattern]) -> str:
    if not text or not mapping:
        return text
    result: List[str] = []
    buffer: List[str] = []
    quote_char: Optional[str] = None
    escape = False

    for ch in text:
        if quote_char:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in ("'", '"'):
            if buffer:
                segment = "".join(buffer)
                result.append(_replace_in_segment(segment, mapping, pattern))
                buffer = []
            result.append(ch)
            quote_char = ch
            continue

        buffer.append(ch)

    if buffer:
        segment = "".join(buffer)
        result.append(_replace_in_segment(segment, mapping, pattern))

    return "".join(result)


def _replace_with_pattern(text: str, mapping: Dict[str, str], pattern: Optional[re.Pattern]) -> str:
    if not text or not mapping or pattern is None:
        return text

    def _repl(match: re.Match) -> str:
        token = match.group("token")
        replacement = mapping.get(token)
        if replacement is None:
            return match.group(0)
        return f"{match.group('prefix')}{replacement}{match.group('suffix')}"

    return pattern.sub(_repl, text)


def strip_cypher_identifier(identifier: str) -> str:
    if not identifier:
        return ""
    value = identifier
    if value.startswith("ROOT__"):
        value = value[len("ROOT__") :]
    if "__" in value:
        value = value.split("__", 1)[1]
    return value


def normalize_cypher_schema(schema: dict, remove_uri: bool = True, remove_foreign_key_attributes: bool = True) -> dict:
    normalized = dict(schema)

    node_props_raw = schema.get("NodeProperties", [])
    if isinstance(node_props_raw, dict):
        node_props_iter = []
        for node_label, props in node_props_raw.items():
            for prop in props:
                entry = dict(prop)
                entry["nodeName"] = node_label
                node_props_iter.append(entry)
    else:
        node_props_iter = [dict(prop) for prop in node_props_raw]

    relationship_tokens: Dict[str, set[str]] = {}
    for rel in schema.get("Relationships", []):
        base = strip_cypher_identifier(rel.get("relationshipType", ""))
        if not base:
            continue
        base_lower = base.lower()
        for label in rel.get("startNodeLabels", []):
            relationship_tokens.setdefault(label, set()).add(base_lower)

    filtered_props: List[dict] = []
    for prop in node_props_iter:
        node_label = prop.get("nodeName")
        prop_name = prop.get("propertyName", "")
        base_name = strip_cypher_identifier(prop_name)
        if not base_name:
            filtered_props.append(prop)
            continue
        base_lower = base_name.lower()
        if remove_uri and base_lower == "uri":
            continue
        if remove_foreign_key_attributes and node_label and base_lower in relationship_tokens.get(node_label, set()):
            continue
        filtered_props.append(prop)

    normalized["NodeProperties"] = filtered_props
    return normalized


@dataclass
class IdentifierMapping:
    forward_by_context: Dict[str, Dict[str, str]] = field(default_factory=dict)
    collisions_by_context: Dict[str, Sequence[Tuple[str, str]]] = field(default_factory=dict)

    def __post_init__(self):
        self.forward_by_context = {ctx: dict(mapping) for ctx, mapping in self.forward_by_context.items()}
        self.collisions_by_context = {ctx: list(collisions) for ctx, collisions in self.collisions_by_context.items()}
        self.reverse_by_context: Dict[str, Dict[str, str]] = {
            ctx: {short: original for original, short in mapping.items()}
            for ctx, mapping in self.forward_by_context.items()
        }
        self._forward_all: Dict[str, str] = {}
        for mapping in self.forward_by_context.values():
            self._forward_all.update(mapping)
        self._reverse_all: Dict[str, str] = {short: original for original, short in self._forward_all.items()}

        self._schema_shorten_pattern = _build_token_pattern(self._forward_all.keys())
        self._schema_restore_pattern = _build_token_pattern(self._reverse_all.keys())

        labels_forward = self.forward_by_context.get(CONTEXT_LABEL, {})
        labels_reverse = self.reverse_by_context.get(CONTEXT_LABEL, {})
        self._label_shorten_pattern = _build_label_pattern(labels_forward.keys())
        self._label_restore_pattern = _build_label_pattern(labels_reverse.keys())

        props_forward = self.forward_by_context.get(CONTEXT_PROPERTY, {})
        props_reverse = self.reverse_by_context.get(CONTEXT_PROPERTY, {})
        self._prop_accessor_shorten_pattern = _build_property_accessor_pattern(props_forward.keys())
        self._prop_accessor_restore_pattern = _build_property_accessor_pattern(props_reverse.keys())
        self._prop_map_shorten_pattern = _build_property_map_pattern(props_forward.keys())
        self._prop_map_restore_pattern = _build_property_map_pattern(props_reverse.keys())

    def shorten_schema(self, text: str) -> str:
        return _replace_outside_string_literals(text, self._forward_all, self._schema_shorten_pattern)

    def restore_schema(self, text: str) -> str:
        return _replace_outside_string_literals(text, self._reverse_all, self._schema_restore_pattern)

    def shorten_query(self, text: str) -> str:
        text = _replace_with_pattern(text, self.forward_by_context.get(CONTEXT_LABEL, {}), self._label_shorten_pattern)
        property_mapping = self.forward_by_context.get(CONTEXT_PROPERTY, {})
        text = _replace_with_pattern(text, property_mapping, self._prop_accessor_shorten_pattern)
        text = _replace_with_pattern(text, property_mapping, self._prop_map_shorten_pattern)
        return text

    def restore_query(self, text: str) -> str:
        text = _replace_with_pattern(text, self.reverse_by_context.get(CONTEXT_LABEL, {}), self._label_restore_pattern)
        property_mapping = self.reverse_by_context.get(CONTEXT_PROPERTY, {})
        text = _replace_with_pattern(text, property_mapping, self._prop_accessor_restore_pattern)
        text = _replace_with_pattern(text, property_mapping, self._prop_map_restore_pattern)
        return text

    # Backwards-compatible aliases
    def shorten(self, text: str) -> str:
        return self.shorten_schema(text)

    def restore(self, text: str) -> str:
        return self.restore_schema(text)

    def to_serializable(self) -> Dict[str, Dict[str, List[str]]]:
        serialized_contexts: Dict[str, Dict[str, List[str]]] = {}
        for ctx, mapping in self.forward_by_context.items():
            originals = list(mapping.keys())
            shorts = [mapping[o] for o in originals]
            serialized_contexts[ctx] = {"original": originals, "short": shorts}
        payload: Dict[str, Dict[str, List[str]] | Dict[str, List[Tuple[str, str]]]] = {"contexts": serialized_contexts}
        collisions_payload = {
            ctx: [(orig, short) for orig, short in collisions]
            for ctx, collisions in self.collisions_by_context.items()
            if collisions
        }
        if collisions_payload:
            payload["collisions"] = collisions_payload
        return payload

    @classmethod
    def from_serializable(cls, data: Dict[str, Dict[str, List[str]]]) -> "IdentifierMapping":
        contexts_payload = data.get("contexts", {})
        forward_by_context: Dict[str, Dict[str, str]] = {}
        for ctx, payload in contexts_payload.items():
            originals = list(payload.get("original", []))
            shorts = list(payload.get("short", []))
            if len(originals) != len(shorts):
                raise ValueError(f"Invalid identifier mapping payload for context `{ctx}`: length mismatch.")
            forward_by_context[ctx] = dict(zip(originals, shorts))

        collisions_payload = {
            ctx: [tuple(item) for item in collisions]
            for ctx, collisions in data.get("collisions", {}).items()
        }
        return cls(forward_by_context=forward_by_context, collisions_by_context=collisions_payload)


class CypherIdentifierMappingBuilder:
    def __init__(self, keep_collisions: bool = True):
        self.keep_collisions = keep_collisions

    def build(self, schema: dict, strategy: str = "none") -> IdentifierMapping:
        strategy = (strategy or "none").lower()
        if strategy == "none":
            return IdentifierMapping()
        if strategy != "strip_prefix":
            raise ValueError(f"Unsupported Cypher identifier mapping strategy `{strategy}`.")

        contexts = self._collect_candidates(schema)
        forward_by_context: Dict[str, Dict[str, str]] = {ctx: {} for ctx in contexts}
        collisions_by_context: Dict[str, List[Tuple[str, str]]] = {ctx: [] for ctx in contexts}
        used_shorts: Dict[str, set] = {ctx: set() for ctx in contexts}

        for ctx, tokens in contexts.items():
            for original in tokens:
                shortened = strip_cypher_identifier(original)
                if not shortened or shortened == original:
                    continue

                if shortened in used_shorts[ctx] and forward_by_context[ctx].get(original) != shortened:
                    collisions_by_context[ctx].append((original, shortened))
                    if self.keep_collisions:
                        continue
                    suffix = 2
                    new_short = f"{shortened}_{suffix}"
                    while new_short in used_shorts[ctx]:
                        suffix += 1
                        new_short = f"{shortened}_{suffix}"
                    shortened = new_short

                forward_by_context[ctx][original] = shortened
                used_shorts[ctx].add(shortened)

        return IdentifierMapping(forward_by_context=forward_by_context, collisions_by_context=collisions_by_context)

    def _collect_candidates(self, schema: dict) -> Dict[str, List[str]]:
        contexts: Dict[str, List[str]] = {
            CONTEXT_LABEL: [],
            CONTEXT_PROPERTY: [],
        }

        contexts[CONTEXT_LABEL].extend(schema.get("NodeLabels", []))
        contexts[CONTEXT_LABEL].extend(schema.get("RelationshipLabels", []))

        for entry in schema.get("NodeProperties", []):
            contexts[CONTEXT_LABEL].append(entry.get("nodeName", ""))
            contexts[CONTEXT_PROPERTY].append(entry.get("propertyName", ""))

        for rel in schema.get("Relationships", []):
            contexts[CONTEXT_LABEL].append(rel.get("relationshipType", ""))
            contexts[CONTEXT_LABEL].extend(rel.get("startNodeLabels", []))
            contexts[CONTEXT_LABEL].extend(rel.get("endNodeLabels", []))

        rel_props = schema.get("RelationshipProperties", [])
        if isinstance(rel_props, dict):
            for rel_name, props in rel_props.items():
                contexts[CONTEXT_LABEL].append(rel_name)
                for prop in props:
                    contexts[CONTEXT_PROPERTY].append(prop.get("propertyName", ""))
        else:
            for entry in rel_props:
                contexts[CONTEXT_LABEL].append(entry.get("relName", ""))
                contexts[CONTEXT_PROPERTY].append(entry.get("propertyName", ""))

        for ctx, tokens in contexts.items():
            seen = set()
            deduped: List[str] = []
            for token in tokens:
                if not token or token in seen:
                    continue
                deduped.append(token)
                seen.add(token)
            contexts[ctx] = deduped

        return contexts
