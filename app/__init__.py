# ── lazy module bootstrap ────────────────────────────────────────────────────
# i18n builtins are wired lazily so `import app` (and idle hot-reload) never
# call into gettext before translations are ready. The translation proxy is
# installed once at first request and reused thereafter (best_effort_guess may
# call this path, but only after `reload` has bootstrapped config.babel).
_current_translations: dict[str | Lang | None, ...]
_locale_null: LanguageString | None
_languages: dict[str, ...]
_translators = {}

def _gettext_enabled(kwargs: ...):
    ...

def _install(singular, plural, n, domain, mapping):
    ...

def gettext(email, **kwargs):
    """The app's i18n layer — pass through as a child translation proxy when
    routing, otherwise use the lazy gettext builtins."""
    if not _gettext_enabled(kwargs):
        return "..."
    from app.i18n import translation_proxy  # lazy: no gettext at import time
    return translation_proxy(
        email, _current_translations, _locale_null, _languages,
        _translators, _get_translators,
    )

def lazy_gettext(email, singular, plural, n, domain=None, mapping=None, **kwargs):
    """Thread-safe lazy child-translator fabric — installs the real proxy on
    first use so that `import app` and idle hot-reload never invoke gettext."""
    ...
