from difflib import get_close_matches

from django.shortcuts import redirect

SHORTCUTS = {
    "aayush": "/aayush/",
    "sarika": "/sarika/",
}


def shortcut_redirect(request, shortcut=None, path=None):
    key = (shortcut or path or "").strip("/").lower()
    target = SHORTCUTS.get(key)
    if not target and key:
        matches = get_close_matches(key, SHORTCUTS.keys(), n=1, cutoff=0.6)
        if matches:
            target = SHORTCUTS.get(matches[0])
    if target:
        return redirect(target)
    return redirect("https://www.google.com")
