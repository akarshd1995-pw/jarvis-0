"""Random text lines that look like what appears on a screen: UI labels, error messages, code, paths, URLs,
chat, numbers, dates. Plus random character strings, so the reader learns letters, not just words."""
from __future__ import annotations

import random
import string

WORDS = """the of and to in is you that it for on with as this was are be at by not from or have an they which
one your all were can there more if will up about out them so what make like time just know take into year
some could see other than then now look only come its over think also back after use two how our work first
well way even new want because any these give day most us error failed invalid password login user account
file folder open save cancel ok yes no settings help view edit tools window close minimize maximize search
home download downloads documents desktop pictures music videos share copy paste delete rename properties
network connection server request response timeout retry status code success warning info debug exception
module import function class return value true false none null undefined object array string number list
install update version release build run test deploy commit push pull merge branch main master origin remote
email message reply forward inbox sent draft attachment meeting calendar today tomorrow yesterday morning
evening price total amount invoice payment receipt order delivery address phone name date time week month
project report budget plan task done pending review approved rejected profile sign out sign in register
forgot reset continue next previous back submit upload page loading please wait connecting connected
offline online available unavailable permission denied access granted allowed blocked restart shutdown
jarvis python java javascript react docker kubernetes linux windows chrome edge youtube github google
train model epoch loss accuracy batch gpu memory cuda tensor dataset validation checkpoint learning rate
hai nahi kya kaise kab kahan karo chalao bhai yaar accha theek""".split()
ERRORS = ["Error: {w} not found", "Login failed: invalid password", "404 Not Found", "500 Internal Server Error",
          "ModuleNotFoundError: No module named '{w}'", "Traceback (most recent call last):",
          "  File \"{p}\", line {n}, in {w}", "TypeError: '{w}' object is not callable", "Access is denied.",
          "Connection timed out after {n} ms", "Permission denied (publickey).", "fatal: not a git repository",
          "SyntaxError: invalid syntax", "KeyError: '{w}'", "ERROR {n}: {W} failed", "Unable to connect to {u}",
          "Your session has expired. Please sign in again.", "Warning: {w} is deprecated", "CUDA out of memory.",
          "npm ERR! code ENOENT", "ValueError: invalid literal for int() with base 10: '{w}'", "Exit code {n}"]
UI = ["File  Edit  View  Help", "Settings", "Sign in", "Forgot password?", "Remember me", "Cancel", "OK", "Save",
      "Search", "Downloads", "New folder", "Recent files", "Share", "Notifications", "Profile", "Log out",
      "Username", "Password", "Email address", "Continue with Google", "Terms of Service", "Privacy Policy",
      "{n} new messages", "Last seen {t}", "Updated {d}", "Showing {n} results", "Page {n} of {m}",
      "{W} - Google Chrome", "{w}.py - Visual Studio Code", "Inbox ({n})", "Battery {n}%", "Wi-Fi connected"]
CODE = ["def {w}({w2}):", "    return {w} + {n}", "import {w}", "from {w} import {w2}", "for i in range({n}):",
        "if {w} is None:", "const {w} = require('{w2}');", "print(f\"{{{w}}}\")", "{w}.{w2}({n})", "}} else {{",
        "SELECT * FROM {w} WHERE id = {n};", "git commit -m \"{w} {w2}\"", "pip install {w}", "npm run {w}",
        "<div class=\"{w}\">", "# TODO: fix {w}", "{w} = [{n}, {m}]", "self.{w} = {w2}", "async function {w}() {{"]


def _w(r: random.Random) -> str:
    return r.choice(WORDS)


def _path(r: random.Random) -> str:
    root = r.choice(["C:\\Users\\akarsh", "C:\\Program Files", "D:\\projects", "/home/user", "~/Downloads"])
    sep = "\\" if root.startswith(("C", "D")) else "/"
    parts = [_w(r) for _ in range(r.randint(1, 3))]
    return root + sep + sep.join(parts) + r.choice(["", ".py", ".txt", ".pdf", ".png", ".json"])


def _url(r: random.Random) -> str:
    return r.choice(["https://", "http://", "www.", ""]) + _w(r) + r.choice([".com", ".in", ".org", ".io", ".dev"]) \
        + r.choice(["", "/" + _w(r), "/" + _w(r) + "/" + str(r.randint(1, 999)), "/?q=" + _w(r)])


def _date(r: random.Random) -> str:
    d, m, y = r.randint(1, 28), r.randint(1, 12), r.randint(2019, 2027)
    return r.choice([f"{d:02d}/{m:02d}/{y}", f"{y}-{m:02d}-{d:02d}", f"{d} {r.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {y}",
                     f"{r.randint(1, 12)}:{r.randint(0, 59):02d} {r.choice(['AM', 'PM'])}"])


def _fill(t: str, r: random.Random) -> str:
    return t.format(w=_w(r), w2=_w(r), W=_w(r).capitalize(), n=r.randint(0, 9999), m=r.randint(0, 999),
                    p=_path(r), u=_url(r), t=_date(r), d=_date(r))


def _case(s: str, r: random.Random) -> str:
    x = r.random()
    return s.capitalize() if x < 0.35 else s.upper() if x < 0.45 else s.title() if x < 0.55 else s


def line(r: random.Random, max_chars: int = 48) -> str:
    x = r.random()
    if x < 0.30:
        s = _case(" ".join(_w(r) for _ in range(r.randint(1, 8))), r)
        if r.random() < 0.3:
            s += r.choice([".", ",", ":", "?", "!", "...", ";"])
    elif x < 0.45:
        s = _fill(r.choice(ERRORS), r)
    elif x < 0.57:
        s = _fill(r.choice(UI), r)
    elif x < 0.69:
        s = _fill(r.choice(CODE), r)
    elif x < 0.75:
        s = _path(r)
    elif x < 0.80:
        s = _url(r)
    elif x < 0.86:
        s = r.choice([_date(r), f"₹{r.randint(1, 99999):,}", f"${r.randint(1, 9999)}.{r.randint(0, 99):02d}",
                      f"{r.randint(1, 100)}%", f"+91 {r.randint(60000, 99999)} {r.randint(10000, 99999)}",
                      f"{r.randint(1, 999)} MB", f"v{r.randint(0, 9)}.{r.randint(0, 20)}.{r.randint(0, 9)}"])
    else:                                            # random characters: learn letters, not a dictionary
        pool = string.ascii_letters + string.digits + string.punctuation + " " * 8
        s = "".join(r.choice(pool) for _ in range(r.randint(3, 24))).strip() or "x"
    s = s.strip()
    if len(s) > max_chars:
        cut = s.rfind(" ", 0, max_chars)
        s = s[: cut if cut > 8 else max_chars].rstrip()
    return s or _w(r)
