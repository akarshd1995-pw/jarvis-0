"""Vocabulary for the data engine.

Closed sets (folders, known apps, system actions) must be fully covered in training.
Open sets (reminder texts, unknown apps, URLs, filenames, queries) are generated
compositionally so the model learns to COPY spans instead of memorizing a list.
"""
from __future__ import annotations

import random

# ---------------------------------------------------------------- folders (closed)
FOLDERS = {
    "Downloads": ["downloads", "Downloads", "my downloads", "the downloads folder", "downloads folder",
                  "download folder", "dl folder", "the download dir", "downloads dir"],
    "Documents": ["documents", "Documents", "my documents", "the documents folder", "docs", "my docs",
                  "documents folder"],
    "Desktop": ["desktop", "Desktop", "my desktop", "the desktop", "desktop screen"],
    "Pictures": ["pictures", "Pictures", "my pictures", "the pictures folder", "photos folder", "pics folder",
                 "images folder"],
    "Music": ["music", "Music", "my music folder", "the music folder", "songs folder"],
    "Videos": ["videos", "Videos", "my videos folder", "the videos folder", "movies folder"],
    "Archive": ["archive", "Archive", "the archive", "the archive folder", "my archive", "archives folder"],
}
PATH_ROOTS = ["C:/Users/me", "D:", "E:", "D:/work", "C:/dev", "D:/backup", "C:/Users/me/OneDrive"]
PATH_PARTS = ["projects", "college", "notes", "invoices", "screenshots", "photos", "old", "2024", "2025",
              "clients", "jarvis", "everway", "assignments", "taxes", "receipts", "music", "datasets",
              "models", "backup", "misc", "temp", "reports", "resume", "travel", "bills"]

# ---------------------------------------------------------------- file types
EXTS = ["pdf", "png", "jpg", "jpeg", "zip", "rar", "7z", "exe", "msi", "docx", "doc", "xlsx", "csv", "pptx",
        "mp4", "mkv", "mov", "mp3", "wav", "txt", "md", "json", "py", "ipynb", "iso", "svg", "gif", "webp",
        "apk", "torrent", "html", "log"]
EXT_NICKNAMES = {"pdf": ["pdfs", "PDFs", "pdf docs"], "jpg": ["jpgs", "jpegs"], "png": ["pngs"],
                 "mp4": ["mp4 videos"], "mp3": ["mp3 songs"], "zip": ["zips", "zip archives"],
                 "exe": ["installers", "exes", "setup files"], "docx": ["word docs", "word files"],
                 "xlsx": ["excel files", "spreadsheets"], "pptx": ["ppts", "powerpoints"],
                 "csv": ["csvs"], "py": ["python files"], "ipynb": ["notebooks"], "apk": ["apks"],
                 "torrent": ["torrent files"], "log": ["log files"]}

# ---------------------------------------------------------------- apps
KNOWN_APPS = {  # alias ident -> surface forms
    "vscode": ["vscode", "VS Code", "vs code", "visual studio code", "code editor", "vsc"],
    "chrome": ["chrome", "Chrome", "google chrome", "the browser", "browser"],
    "edge": ["edge", "microsoft edge", "MS edge"],
    "notepad": ["notepad", "Notepad", "note pad"],
    "explorer": ["file explorer", "explorer", "file manager", "my computer"],
    "terminal": ["terminal", "the terminal", "windows terminal", "cmd", "command prompt", "powershell"],
    "spotify": ["spotify", "Spotify", "music app"],
    "calculator": ["calculator", "calc"],
    "settings": ["settings", "windows settings", "system settings"],
}
OTHER_APPS = ["Obsidian", "Discord", "Slack", "Zoom", "Teams", "WhatsApp", "Telegram", "Steam", "Figma",
              "Notion", "Postman", "Docker Desktop", "Blender", "Unity Hub", "OBS", "VLC", "Photoshop",
              "Word", "Excel", "PowerPoint", "Outlook", "Firefox", "Brave", "Opera", "GitHub Desktop",
              "PyCharm", "IntelliJ", "Android Studio", "Paint", "Snipping Tool", "Task Manager", "Anaconda",
              "Jupyter", "Cursor", "Arc", "Epic Games", "Audacity", "GIMP", "Krita", "Canva", "Thunderbird",
              "Signal", "Skype", "Evernote", "Todoist", "MATLAB", "Arduino IDE", "Wireshark", "Putty",
              "WinRAR", "7-Zip", "Kindle", "Netflix", "Prime Video", "Hotstar", "JioCinema"]

# ---------------------------------------------------------------- urls
KNOWN_SITES = {"youtube": "youtube.com", "github": "github.com", "gmail": "mail.google.com",
               "leetcode": "leetcode.com", "kaggle": "kaggle.com", "linkedin": "linkedin.com",
               "stack overflow": "stackoverflow.com", "hugging face": "huggingface.co",
               "vercel": "vercel.com", "hacker news": "news.ycombinator.com", "chatgpt": "chatgpt.com",
               "claude": "claude.ai", "reddit": "reddit.com", "twitter": "x.com", "amazon": "amazon.in",
               "flipkart": "flipkart.com", "netflix": "netflix.com", "google drive": "drive.google.com",
               "google calendar": "calendar.google.com", "wikipedia": "wikipedia.org", "irctc": "irctc.co.in",
               "swiggy": "swiggy.com", "zomato": "zomato.com", "codeforces": "codeforces.com"}
URL_WORDS = ["pixel", "nova", "byte", "cloud", "green", "kite", "lotus", "metro", "quick", "smart", "tech",
             "learn", "code", "shop", "news", "daily", "craft", "data", "astro", "ocean", "rapid", "tiny",
             "bright", "urban", "spice", "chai", "desi", "maker", "lab", "hub"]
TLDS = [".com", ".in", ".io", ".dev", ".ai", ".org", ".net", ".app", ".co", ".co.in"]

# ---------------------------------------------------------------- reminders (open)
R_VERBS = ["call", "text", "email", "message", "meet", "pay", "submit", "review", "finish", "start", "check",
           "book", "buy", "order", "return", "renew", "cancel", "send", "push", "deploy", "backup", "clean",
           "water", "feed", "pick up", "drop off", "charge", "update", "prepare", "practice", "revise",
           "read", "watch", "fix", "test", "reply to", "follow up with", "wish", "schedule", "print"]
R_OBJECTS = ["mom", "dad", "the landlord", "my manager", "the team", "the electricity bill", "the rent",
             "the assignment", "the report", "the PR", "my code", "the server", "the laptop", "the plants",
             "the cat", "the dog", "groceries", "medicines", "the train ticket", "the flight", "the library book",
             "the passport", "the insurance", "the subscription", "the invoice", "the slides", "the resume",
             "the demo", "the presentation", "DSA", "the model training", "the dataset", "the database",
             "the portfolio site", "the Everway build", "the phone", "the client", "the interview", "the doctor",
             "the gym session", "the milk", "the course", "the exam notes", "my friend", "the birthday cake"]
NAMES = ["Rahul", "Priya", "Aman", "Sneha", "Arjun", "Neha", "Vikram", "Ananya", "Rohan", "Kavya", "Ishaan",
         "Meera", "Karan", "Divya", "Sam", "Alex", "John", "Sara", "Riya", "Aditya", "Pooja", "Nikhil"]
R_STANDALONE = ["drink water", "stretch", "take a break", "sleep", "stand up", "go for a walk", "meditate",
                "take my medicine", "join the standup", "leave for office", "start the training run",
                "turn off the geyser", "lock the door", "eat lunch", "log my hours", "close the window"]


def remind_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        return rng.choice(R_STANDALONE)
    if r < 0.45:
        return f"{rng.choice(['call', 'text', 'meet', 'email', 'wish', 'reply to'])} {rng.choice(NAMES)}"
    return f"{rng.choice(R_VERBS)} {rng.choice(R_OBJECTS)}"

# ---------------------------------------------------------------- filenames (open)
NAME_WORDS = ["invoice", "resume", "screenshot", "report", "bill", "ticket", "draft", "notes", "assignment",
              "receipt", "final", "backup", "scan", "statement", "salary", "offer", "certificate", "photo",
              "setup", "lecture", "project", "budget", "contract", "aadhaar", "pan", "marksheet", "copy"]


def name_fragment(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return rng.choice(NAME_WORDS)
    if r < 0.75:
        return f"{rng.choice(NAME_WORDS)}_{rng.choice(['v2', 'final', '2024', '2025', 'old', 'new', '01'])}"
    if r < 0.9:
        return rng.choice(["IMG_", "Screenshot", "WhatsApp Image", "Q1", "Q2", "Q3", "Q4", "DSC", "VID_"])
    return f"{rng.choice(NAMES).lower()}"

# ---------------------------------------------------------------- memory search (open)
Q_ADJ = ["", "", "the blue", "that", "the old", "the latest", "my", "the", "the long", "the shared"]
Q_NOUN = ["error message", "chart", "screenshot", "invoice", "python snippet", "meeting notes", "API key docs",
          "train ticket pdf", "resume", "receipt", "recipe", "lecture slides", "bank statement", "design mockup",
          "SQL query", "email draft", "budget sheet", "offer letter", "tax form", "link Rahul sent"]
Q_TAIL = ["", "", "", "from yesterday", "from last week", "I copied", "about GST", "about the server",
          "with the graph", "from the meeting", "I saved last month", "for the project"]


def search_query(rng: random.Random) -> str:
    return " ".join(p for p in [rng.choice(Q_ADJ), rng.choice(Q_NOUN), rng.choice(Q_TAIL)] if p)

# ---------------------------------------------------------------- system (closed)
SYSTEM = {
    "lock": ["lock the screen", "lock my pc", "lock the computer", "lock", "lock my laptop", "lock it",
             "screen lock kar do", "pc lock karo", "lock kar do", "i'm stepping away, lock the pc"],
    "mute": ["mute", "mute the sound", "mute audio", "mute my laptop", "silence the speakers", "sound off",
             "sound band karo", "awaaz band kar do", "mute kar do"],
    "volume_up": ["volume up", "increase the volume", "make it louder", "turn it up", "louder",
                  "raise the volume", "awaaz badhao", "volume badha do", "thoda loud karo"],
    "volume_down": ["volume down", "decrease the volume", "make it quieter", "turn it down", "quieter",
                    "lower the volume", "awaaz kam karo", "volume kam kar do", "thoda dheere karo"],
    "sleep_display": ["turn off the display", "sleep the monitor", "screen off", "turn off the screen",
                      "switch off the monitor", "screen off kar do", "display band karo"],
}

# ---------------------------------------------------------------- out of scope -> unknown()
UNKNOWN_TEMPLATES = [
    "what's the weather {when}", "write a poem about {topic}", "tell me a joke", "who are you",
    "how are you", "what is {topic}", "explain {topic}", "send an email to {name}", "order {food} from swiggy",
    "book a cab to {place}", "play {song}", "what time is it", "translate {topic} to hindi",
    "write python code for {topic}", "summarize this article", "call {name}", "text {name} that i'm late",
    "buy {thing} on amazon", "turn on the lights", "set the ac to 24", "how much is bitcoin",
    "who won the match {when}", "recommend a movie", "what should i eat", "do my homework",
    "hack my neighbor's wifi", "format the c drive", "delete system32", "shut down everyone's pc",
    "make me a website", "what's on my calendar {when}", "how do i {topic}", "i love you", "thanks",
    "hello", "good morning", "good night", "ok", "hmm", "nothing", "never mind", "asdfgh", "?",
    "kya haal hai", "mausam kaisa hai", "ek joke sunao", "mera homework kar do", "{topic} kya hai",
    "khana order karo", "{name} ko call karo", "gaana bajao", "tum kaun ho", "kuch nahi",
    "move my stuff", "delete it", "put them somewhere", "remind me", "open", "copy", "do the thing",
]
U_FILL = {
    "when": ["today", "tomorrow", "this weekend", "in delhi", "right now", "yesterday", "tonight"],
    "topic": ["the moon", "machine learning", "quantum physics", "love", "chai", "black holes", "recursion",
              "my college", "transformers", "inflation", "cricket", "bake a cake", "fix my wifi", "learn rust"],
    "name": NAMES, "food": ["biryani", "pizza", "dosa", "momos", "a burger"],
    "place": ["the airport", "koramangala", "the station", "office"],
    "song": ["some lofi", "arijit singh", "the latest hits", "kesariya"],
    "thing": ["a mouse", "headphones", "a keyboard", "a book"],
}


def unknown_text(rng: random.Random) -> str:
    t = rng.choice(UNKNOWN_TEMPLATES)
    for k, vals in U_FILL.items():
        t = t.replace("{" + k + "}", rng.choice(vals))
    return t

# ---------------------------------------------------------------- numbers
EN_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
EN_TENS = {30: "thirty", 40: "forty", 45: "forty five", 50: "fifty", 60: "sixty", 90: "ninety"}
HI_NUM = {1: "ek", 2: "do", 3: "teen", 4: "chaar", 5: "paanch", 6: "chhe", 7: "saat", 8: "aath", 9: "nau",
          10: "das", 12: "baarah", 15: "pandrah", 20: "bees", 25: "pachchees", 30: "tees", 40: "chaalis",
          45: "paintalis", 50: "pachaas"}


def num_surface(rng: random.Random, n: int, hinglish: bool = False) -> str:
    if hinglish and n in HI_NUM and rng.random() < 0.5:
        return HI_NUM[n]
    if rng.random() < 0.25:
        if n <= 20:
            return EN_NUM[n]
        if n in EN_TENS:
            return EN_TENS[n]
    return str(n)
