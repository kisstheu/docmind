# docmind-domain-sdk

`docmind-domain-sdk` is the neutral, independently installable Python SDK for
DocMind domain plugin protocol 1.1. It contains only JSON-serializable DTOs, the
asynchronous `DomainPlugin` port, cross-message boundary validation, and
versioned JSON Schema bundles. It does not depend on DocMind Core.

The Python package version and wire protocol version evolve independently. This
release is SDK `0.2.0` and protocol `1.1`.

## Request-local options

`DomainRequest.options` is a request-local `dict[str, JsonValue]`. It defaults
to an empty object and accepts only recursive JSON values: objects with string
keys, arrays, strings, booleans, null, integers, and finite numbers. The SDK
copies the complete value tree during request construction, so later mutations
of caller-owned dictionaries or lists do not change the request.

The SDK and Host validate only the neutral JSON and resource-boundary contract.
The Host transfers options without interpreting, filtering, renaming, or adding
values. Each plugin owns its option keys, business validation, unknown-key
policy, and conversion to internal types. Runtime objects such as connections,
loggers, paths, callbacks, and credentials must not enter this DTO.

Every options tree is limited to:

- 16,384 UTF-8 bytes using the protocol's compact, sorted JSON encoding;
- container depth 8, counting the top-level options object as depth 1;
- 128 object keys across the complete tree;
- 256 combined object members and array elements across the complete tree.

`StaticDomainHost` repeats this neutral validation immediately before creating
the plugin `execute` coroutine. Errors report only a fixed failure category and
limit; option keys and values are not included.

## Protocol 1.0 compatibility

The frozen `protocol-1.0.schema.json` artifact remains in the package unchanged.
SDK 0.2.0 can read a protocol 1.0 `DomainRequest` only when the wire payload
omits `options`. An explicitly supplied `options` field is invalid for 1.0 even
when it is `{}`. A valid legacy request is normalized to `request.options == {}`
in memory, while `model_dump(mode="json")` and `model_dump_json()` omit that
field again so the result remains valid protocol 1.0 wire data.

All other top-level DTOs created by this release use protocol 1.1. New 1.1 wire
requests are not promised to be readable by old 1.0 receivers.

## Content and identifier rules

`SourceSnapshot.inline_text` is preserved character-for-character. Its
`content_sha256` is calculated from the exact UTF-8 bytes of that string,
without trimming, newline conversion, or Unicode normalization. For
URI-backed content, `content_sha256` describes the actual raw bytes fetched
from the URI; the SDK does not fetch the resource itself.

Protocol identifiers reject leading or trailing whitespace instead of silently
trimming it. URI values are constrained only by non-empty value
and maximum length; no URI scheme allowlist is defined.

`PluginStartRequest.config` accepts JSON-serializable values. Nested values are
not promised to be deeply immutable.

## Development

From the repository root:

```bash
.venv/bin/python -m pytest -q packages/docmind-domain-sdk/tests
.venv/bin/python packages/docmind-domain-sdk/scripts/generate_json_schema.py --check
```
