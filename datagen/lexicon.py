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
EXT_NICKNAMES = {"pdf": ["pdfs", "PDFs", "pdf docs"], "jpg": ["jpgs"], "jpeg": ["jpegs"], "png": ["pngs"],
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

# ---------------------------------------------------------------- made-up words
# Random pronounceable words force the model to COPY names character by character instead of
# recalling a familiar word (it used to turn D:/jarvis/reports into D:/reports/jarvis).
_ONSETS = ["b", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "sh", "t", "v", "y", "z",
           "br", "kr", "tr", "pr", "st", "bh", "dh", "gh", "kh", "th"]
_VOWELS = ["a", "e", "i", "o", "u", "aa", "ee", "oo", "ai", "ou"]
_CODAS = ["", "", "", "n", "r", "l", "s", "t", "k", "m", "x"]


def pseudo_word(rng: random.Random) -> str:
    w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(rng.randint(1, 3))) + rng.choice(_CODAS)
    r = rng.random()
    if r < 0.12:
        w += str(rng.randint(1, 2099))
    elif r < 0.2:
        w += rng.choice(["_", "-"]) + rng.choice(["v2", "final", "old", "new", "x", "pro", str(rng.randint(1, 99))])
    if rng.random() < 0.1:
        w = w.capitalize()
    return w


def path_part(rng: random.Random) -> str:
    return rng.choice(PATH_PARTS) if rng.random() < 0.5 else pseudo_word(rng)


def url_name(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.35:
        return "".join(rng.sample(URL_WORDS, rng.choice([1, 2])))
    if r < 0.8:
        return pseudo_word(rng).lower().replace("_", "-")
    return rng.choice(URL_WORDS) + pseudo_word(rng).lower().replace("_", "-")

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
    if r < 0.2:
        return pseudo_word(rng)
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
UNKNOWN_TEMPLATES = [  # chit-chat, questions, unclear or incomplete requests -> unknown()
    "what's the weather {when}", "write a poem about {topic}", "tell me a joke", "who are you",
    "how are you", "what is {topic}", "explain {topic}", "what time is it", "translate {topic} to hindi",
    "write python code for {topic}", "summarize this article", "how much is bitcoin",
    "who won the match {when}", "recommend a movie", "what should i eat", "do my homework",
    "make me a website", "what's on my calendar {when}", "how do i {topic}", "i love you", "thanks",
    "hello", "good morning", "good night", "ok", "hmm", "nothing", "never mind", "asdfgh", "?",
    "kya haal hai", "mausam kaisa hai", "ek joke sunao", "mera homework kar do", "{topic} kya hai",
    "gaana bajao", "tum kaun ho", "kuch nahi", "move my stuff", "delete it", "put them somewhere",
    "remind me", "open", "copy", "do the thing", "play", "search", "play something",
    "what can you do", "can you delete files", "are you able to move files", "do you know {topic}",
    "help", "what are your features", "tum kya kya kar sakte ho", "test", "testing 123", "lol", "bruh",
]
# Clear actions Jarvis cannot do yet -> unsupported(request="...") (also inside multi-step commands)
UNSUPPORTED_TEMPLATES = [
    "send an email to {name}", "email {name} the report", "call {name}", "text {name} that i'm late",
    "message {name} on whatsapp", "book a cab to {place}", "order {food} from swiggy", "buy {thing} on amazon",
    "turn on the lights", "set the ac to 24", "pay the electricity bill", "post this on instagram",
    "connect to the office wifi", "pair my headphones", "install {app}", "uninstall {app}", "update windows",
    "empty the recycle bin", "change my wallpaper", "turn on night light",
    "format the c drive", "delete system32", "hack my neighbor's wifi", "shut down everyone's pc",
    "{name} ko call karo", "{name} ko message bhejo", "khana order karo", "cab book karo",
    "light on karo", "{app} install karo", "office wifi se connect karo",
    "share this file with {name}", "print the document", "zip the downloads folder", "rename the file",
]
U_FILL_APP = ["discord", "vlc", "steam", "python", "node", "zoom", "figma"]
U_FILL = {
    "when": ["today", "tomorrow", "this weekend", "in delhi", "right now", "yesterday", "tonight"],
    "topic": ["the moon", "machine learning", "quantum physics", "love", "chai", "black holes", "recursion",
              "my college", "transformers", "inflation", "cricket", "bake a cake", "fix my wifi", "learn rust"],
    "name": NAMES, "food": ["biryani", "pizza", "dosa", "momos", "a burger"],
    "place": ["the airport", "koramangala", "the station", "office"],
    "song": ["some lofi", "arijit singh", "the latest hits", "kesariya"],
    "thing": ["a mouse", "headphones", "a keyboard", "a book"],
}


def _fill(t: str, rng: random.Random) -> str:
    for k, vals in {**U_FILL, "app": U_FILL_APP}.items():
        t = t.replace("{" + k + "}", rng.choice(vals))
    return t


def unknown_text(rng: random.Random) -> str:
    return _fill(rng.choice(UNKNOWN_TEMPLATES), rng)


def unsupported_text(rng: random.Random) -> str:
    return _fill(rng.choice(UNSUPPORTED_TEMPLATES), rng)


# ---------------------------------------------------------------- web search (open queries)
SEARCH_SITES = {
    "youtube": ["youtube", "YouTube", "yt", "you tube"],
    "google": ["google", "Google", "the web", "the internet"],
    "amazon": ["amazon", "Amazon"],
    "flipkart": ["flipkart", "Flipkart"],
    "swiggy": ["swiggy", "Swiggy"],
    "zomato": ["zomato", "Zomato"],
    "github": ["github", "GitHub"],
    "stackoverflow": ["stack overflow", "stackoverflow", "Stack Overflow"],
    "wikipedia": ["wikipedia", "wiki", "Wikipedia"],
    "spotify": ["spotify", "Spotify"],
    "maps": ["google maps", "maps", "Maps"],
    "reddit": ["reddit", "Reddit"],
    "linkedin": ["linkedin", "LinkedIn"],
}
SONGS = ["dear god", "lofi beats", "tum hi ho", "kesariya", "shape of you", "believer", "blinding lights",
         "night changes", "perfect", "bohemian rhapsody", "chaiyya chaiyya", "apna bana le", "heeriye",
         "calm down", "starboy", "photograph", "levitating", "the nights", "faded", "senorita"]
ARTISTS = ["arijit singh", "ap dhillon", "coldplay", "taylor swift", "the weeknd", "ed sheeran", "pritam",
           "diljit dosanjh", "shreya ghoshal", "imagine dragons", "linkin park", "avicii", "badshah", "anuv jain"]
GENRES = ["lofi", "chill", "workout", "study", "bollywood", "punjabi", "jazz", "rain sounds", "focus", "90s hindi"]
HOWTO = ["fix wifi on windows", "make biryani", "tie a tie", "learn python", "train a transformer", "solve a rubik's cube",
         "use git rebase", "improve handwriting", "make cold coffee", "set up a react app", "center a div"]
TOPICS = ["black holes", "transformers", "the french revolution", "quantum computing", "gst rules", "photosynthesis",
          "the mughal empire", "kubernetes", "inflation", "ipl 2026 schedule", "rtx 5070 benchmarks"]
PRODUCTS = ["wireless mouse", "mechanical keyboard", "usb c hub", "laptop stand", "noise cancelling headphones",
            "ssd 1tb", "monitor 27 inch", "phone case", "water bottle", "running shoes", "power bank", "webcam"]
FOODS = ["biryani", "pizza", "masala dosa", "momos", "paneer tikka", "chole bhature", "burger", "shawarma",
         "pav bhaji", "ice cream", "thali"]
CODE = ["python list comprehension", "pytorch cuda out of memory", "react useEffect cleanup", "rust borrow checker",
        "git merge conflict", "numpy broadcasting error", "fastapi cors", "docker compose example",
        "typescript generics", "sql join example", "beam search implementation"]
PLACES = ["petrol pump near me", "cafes near me", "atm near me", "hospital near me", "metro station near me",
          "koramangala", "india gate", "nearest pharmacy", "gyms in indiranagar", "pune railway station"]


def web_query(rng: random.Random, site: str) -> str:
    r = rng.random()
    if site in ("youtube", "spotify"):
        if site == "youtube" and r < 0.25:
            return rng.choice([f"how to {rng.choice(HOWTO)}", f"{rng.choice(TOPICS)} explained",
                               f"{rng.choice(TOPICS)} tutorial", "cricket highlights", "funny cat videos"])
        if r < 0.55:
            return rng.choice(SONGS) + rng.choice(["", "", " song", " lyrics", " live", " remix"])
        if r < 0.8:
            return rng.choice([f"{rng.choice(ARTISTS)} songs", rng.choice(ARTISTS), f"{rng.choice(SONGS)} {rng.choice(ARTISTS)}"])
        return rng.choice([f"{rng.choice(GENRES)} music", f"{rng.choice(GENRES)} mix", f"{rng.choice(GENRES)} playlist",
                           pseudo_word(rng) + " song"])
    if site in ("amazon", "flipkart"):
        p = rng.choice(PRODUCTS)
        return rng.choice([p, f"{p} under {rng.choice([500, 999, 1500, 2000, 5000])}", f"best {p}", f"{pseudo_word(rng)} {p}"])
    if site in ("swiggy", "zomato"):
        return rng.choice(FOODS) + rng.choice(["", "", " near me", " delivery", " under 200"])
    if site in ("github", "stackoverflow"):
        return rng.choice(CODE) if r < 0.8 else f"{pseudo_word(rng).lower()} {rng.choice(['library', 'repo', 'error'])}"
    if site == "maps":
        return rng.choice(PLACES)
    if site in ("wikipedia",):
        return rng.choice(TOPICS)
    pool = [rng.choice(TOPICS), f"how to {rng.choice(HOWTO)}", rng.choice(CODE), rng.choice(PRODUCTS) + " price",
            f"{rng.choice(FOODS)} recipe", rng.choice(PLACES), f"what is {pseudo_word(rng).lower()}"]
    return rng.choice(pool)

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


# ---------------------------------------------------------------- media (vast): creators, songs, artists, shows
CREATORS = [
    "Yes Theory", "301 Diaries", "MKBHD", "Marques Brownlee", "Linus Tech Tips", "Veritasium", "Kurzgesagt", "3Blue1Brown",
    "Fireship", "Two Minute Papers", "Computerphile", "Mrwhosetheboss", "Unbox Therapy", "Dave2D", "MrBeast",
    "Dude Perfect", "Mark Rober", "Vsauce", "Tom Scott", "Johnny Harris", "Vox", "Wendover Productions", "Ali Abdaal",
    "Casey Neistat", "Peter McKinnon", "Mrwhosetheboss", "Lofi Girl", "TED", "TEDx Talks", "BBC News", "WION",
    "CarryMinati", "BB Ki Vines", "Ashish Chanchlani", "Dhruv Rathee", "Technical Guruji", "Trakin Tech", "BeerBiceps",
    "Mostly Sane", "Zakir Khan", "Tanmay Bhat", "Samay Raina", "Physics Wallah", "Apna College", "CodeWithHarry",
    "Striver", "Love Babbar", "Mumbiker Nikhil", "Sourav Joshi Vlogs", "Flying Beast", "Ankur Warikoo", "Think School",
    "Abhi and Niyu", "Bhuvan Bam", "The Ranveer Show", "Nischa", "Kunal Shah", "Andrej Karpathy", "Yannic Kilcher",
    "StatQuest", "sentdex", "Corey Schafer", "Traversy Media", "The Primeagen", "Theo", "NetworkChuck", "Ben Eater",
    "Sebastian Lague", "Brackeys", "Code Bullet", "Stuff Made Here", "Michael Reeves", "Smarter Every Day",
    "Real Engineering", "Lemmino", "Kitboga", "Matt D'Avella", "Nas Daily", "Drew Binsky", "Mark Wiens",
    "Gaurav Taneja", "Harsh Beniwal", "Round2Hell", "Elvish Yadav", "Triggered Insaan", "Fukra Insaan",
    "Slayy Point", "Prajakta Koli", "Kabita's Kitchen", "Ranveer Brar", "Nisha Madhulika", "Gordon Ramsay",
    "Joshua Weissman", "Binging with Babish", "Formula 1", "ICC", "IPL", "ESPNcricinfo", "Netflix India", "T-Series",
]
CREATOR_SUFFIX = ["Vlogs", "Tech", "Diaries", "TV", "Official", "Studios", "Podcast", "Gaming", "Kitchen", "Talks",
                  "Explained", "Live", "Shorts", "Academy", "Films", "Music"]


def creator(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.65:
        return rng.choice(CREATORS)
    if r < 0.85:
        return f"{pseudo_word(rng).capitalize()} {rng.choice(CREATOR_SUFFIX)}"
    if r < 0.93:
        return f"{rng.randint(2, 999)} {rng.choice(CREATOR_SUFFIX)}"
    return pseudo_word(rng).capitalize()


SONGS_MORE = [
    "dear god", "tum hi ho", "kesariya", "shape of you", "believer", "blinding lights", "night changes", "perfect",
    "bohemian rhapsody", "chaiyya chaiyya", "apna bana le", "heeriye", "calm down", "starboy", "photograph",
    "levitating", "the nights", "faded", "senorita", "channa mereya", "raataan lambiyan", "kal ho naa ho", "agar tum saath ho",
    "tere vaaste", "satranga", "pasoori", "jhoome jo pathaan", "naatu naatu", "srivalli", "maan meri jaan", "excuses",
    "brown munde", "295", "lover", "summer high", "o maahi", "chaleya", "husn", "baarishein", "alag aasmaan",
    "bairan", "jo tum mere ho", "hotel california", "stairway to heaven", "numb", "in the end", "yellow", "fix you",
    "viva la vida", "see you again", "counting stars", "radioactive", "thunder", "demons", "closer", "something just like this",
    "anti-hero", "cruel summer", "flowers", "as it was", "watermelon sugar", "die for you", "save your tears", "cheap thrills",
    "unstoppable", "attention", "let her go", "riptide", "heat waves", "stay", "peaches", "drivers license",
    "espresso", "apt", "die with a smile", "birds of a feather", "golden hour", "until i found you", "lag ja gale",
    "pehla nasha", "tujhe dekha toh", "gerua", "zaalima", "hawayein", "khairiyat", "shayad", "bekhayali", "kabira",
    "ilahi", "tera ban jaunga", "dil diyan gallan", "malang sajna", "aaj ki raat", "tauba tauba", "sajni", "ishq",
]
ARTISTS_MORE = [
    "arijit singh", "ap dhillon", "coldplay", "taylor swift", "the weeknd", "ed sheeran", "pritam", "diljit dosanjh",
    "shreya ghoshal", "imagine dragons", "linkin park", "avicii", "badshah", "anuv jain", "karan aujla", "sidhu moose wala",
    "atif aslam", "sonu nigam", "kishore kumar", "lata mangeshkar", "a r rahman", "vishal mishra", "jubin nautiyal",
    "king", "divine", "seedhe maut", "prateek kuhad", "local train", "the beatles", "queen", "eagles", "eminem",
    "drake", "billie eilish", "dua lipa", "bruno mars", "sabrina carpenter", "olivia rodrigo", "harry styles",
    "post malone", "justin bieber", "lady gaga", "kendrick lamar", "travis scott", "bts", "blackpink", "rosé",
    "sunidhi chauhan", "neha kakkar", "mohit chauhan", "kk", "shankar mahadevan", "anirudh", "sid sriram",
]
GENRES_MORE = ["lofi", "lofi hip hop", "chill", "workout", "gym", "study", "focus", "deep focus", "bollywood", "punjabi",
               "jazz", "rain sounds", "90s hindi", "old hindi", "romantic hindi", "sad songs", "party", "edm", "classical",
               "instrumental", "piano", "sleep", "meditation", "coding", "synthwave", "rock", "metal", "indie", "sufi",
               "ghazal", "devotional", "bhajan", "k-pop", "hip hop", "rap", "road trip", "morning", "acoustic", "chill vibes"]
SONG_EXTRA = ["", "", "", "", " song", " lyrics", " remix", " slowed reverb", " 8d audio", " acoustic", " cover",
              " official video", " full song", " audio"]
SHOWS = ["the boys", "stranger things", "money heist", "panchayat", "mirzapur", "the family man", "sacred games",
         "breaking bad", "game of thrones", "house of the dragon", "squid game", "wednesday", "the office", "friends",
         "kota factory", "scam 1992", "dune part two", "oppenheimer", "interstellar", "jawan", "pushpa 2", "kalki 2898 ad",
         "animal", "deadpool and wolverine", "inside out 2", "avatar 3", "the batman", "spider-man", "kgf chapter 2",
         "stree 2", "rrr", "12th fail", "laapataa ladies"]
PODCASTS = ["the ranveer show", "lex fridman podcast", "huberman lab", "joe rogan experience", "figuring out",
            "the diary of a ceo", "acquired", "lenny's podcast", "dostcast", "the bombay journey", "think school podcast"]
LIVE_THINGS = ["lofi girl", "nasa", "isro launch", "ipl match", "aaj tak", "bbc news", "abp news", "dd news", "ndtv",
               "wion", "lofi hip hop radio", "synthwave radio", "chill radio", "jazz radio", "f1 race", "spacex launch",
               "cricket score", "sky news", "al jazeera english"]


# Regional and less-known titles: names the model must copy whole, word by word
SONGS_REGIONAL = [
    "super machi", "vaathi coming", "arabic kuthu", "why this kolaveri di", "rowdy baby", "kutti story", "naatu naatu",
    "oo antava", "srivalli", "butta bomma", "ramuloo ramulaa", "samajavaragamana", "inkem inkem", "kalaavathi",
    "jimikki kammal", "malare", "entammede jimikki", "kannu kannu", "belageddu", "bombe helutaite", "chuttamalle",
    "hukum", "jailer theme", "naa ready", "badass", "kaavaalaa", "ranjithame", "jalabulajangu", "aaluma doluma",
    "maari thara local", "en jeevan", "vaa vaathi", "munbe vaa", "nenjukkul peidhidum", "kanimozhiye", "otha sollaala",
    "ainvayi ainvayi", "lehanga", "brown munde", "excuses", "insane", "with you", "softly", "335", "52 bars",
    "mann mera", "maiyya mainu", "pasoori", "kahani suno", "jhol", "tu hai kahan", "sajni", "o maahi", "husn",
    "tauba tauba", "aaj ki raat", "khalasi", "dil tu jaan tu", "big dawgs", "millionaire", "winning speech",
]
# Common words that are also the start of longer names ("sup" -> "super machi"): copy all of it
OPEN_WORDS = ["super", "sup", "hey", "ok", "yo", "hi", "top", "big", "old", "my", "one", "two", "red", "blue",
              "baby", "love", "night", "dil", "mera", "tera", "jaan", "raja", "rani", "machi", "thala", "anna",
              "pyaar", "ishq", "kanne", "nanba", "chinna", "maa", "bro", "boss", "king", "queen", "street", "local",
              "mass", "theme", "bgm", "song", "dance", "party", "rap", "beat", "vibe", "mood", "golden", "silver"]


def open_query(rng: random.Random) -> str:
    """1-4 words mixing real words and never-seen pseudo-words, so the model copies any name in full."""
    n = rng.choice([1, 2, 2, 2, 3, 3, 4])
    words = [rng.choice(OPEN_WORDS) if rng.random() < 0.45 else pseudo_word(rng).lower().strip("_-") for _ in range(n)]
    if rng.random() < 0.25:
        words.append(rng.choice(["song", "bgm", "theme", "remix", "trailer", "lyrics", "full song", "cover"]))
    return " ".join(w for w in words if w) or pseudo_word(rng)


def song_query(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.2:
        return rng.choice(SONGS_REGIONAL) + rng.choice(SONG_EXTRA)
    if r < 0.42:
        return open_query(rng)
    if r < 0.5:
        return rng.choice(SONGS_MORE) + rng.choice(SONG_EXTRA)
    if r < 0.7:
        return f"{rng.choice(SONGS_MORE)} by {rng.choice(ARTISTS_MORE)}"
    if r < 0.82:
        return f"{rng.choice(ARTISTS_MORE)} {rng.choice(['songs', 'hits', 'new song', 'best songs', 'top songs'])}"
    if r < 0.9:
        return rng.choice(ARTISTS_MORE)
    return f"{pseudo_word(rng)} {rng.choice(['song', 'track', 'remix'])}"


def video_query(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.25:
        return f"{rng.choice(SHOWS)} trailer"
    if r < 0.45:
        return f"how to {rng.choice(HOWTO)}"
    if r < 0.6:
        return f"{rng.choice(TOPICS)} {rng.choice(['explained', 'tutorial', 'documentary', 'in 10 minutes'])}"
    if r < 0.72:
        return rng.choice(["funny cat videos", "cricket highlights", "ipl highlights", "f1 highlights",
                           "stand up comedy", "car reviews", "phone unboxing", "travel vlog", "street food india"])
    if r < 0.82:
        return f"{rng.choice(PODCASTS)} {rng.choice(['', 'episode', 'best episode', 'clips'])}".strip()
    return f"{rng.choice(CREATORS)} {rng.choice(['best moments', 'challenge', 'review', 'vlog', 'interview'])}"
