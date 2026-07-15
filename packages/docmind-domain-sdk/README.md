# docmind-domain-sdk

`docmind-domain-sdk` is the neutral, independently installable Python SDK for
DocMind domain plugin protocol 1.0. It contains only JSON-serializable DTOs, the
asynchronous `DomainPlugin` port, cross-message boundary validation, and the
versioned JSON Schema bundle. It does not depend on DocMind Core.

The Python package version and wire protocol version evolve independently. This
release is SDK `0.1.0` and protocol `1.0`.

## Content and identifier rules

`SourceSnapshot.inline_text` is preserved character-for-character. Its
`content_sha256` is calculated from the exact UTF-8 bytes of that string,
without trimming, newline conversion, or Unicode normalization. For
URI-backed content, `content_sha256` describes the actual raw bytes fetched
from the URI; the SDK does not fetch the resource itself.

Protocol identifiers reject leading or trailing whitespace instead of silently
trimming it. URI values in protocol 1.0 are constrained only by non-empty value
and maximum length; no URI scheme allowlist is defined.

`PluginStartRequest.config` accepts JSON-serializable values. Nested values are
not promised to be deeply immutable.

## Development

From the repository root:

```bash
.venv/bin/python -m pytest -q packages/docmind-domain-sdk/tests
.venv/bin/python packages/docmind-domain-sdk/scripts/generate_json_schema.py --check
```
