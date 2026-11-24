"""Directions: "fastest route from Koramangala to the airport", "how long to drive home", "office kitna door hai".

Always opens Google Maps with the route (fastest route + live traffic on screen). Jarvis also says the time:
  - Google Routes API (live traffic) when you've put your own API key in Settings > Places & maps
  - otherwise an OpenStreetMap estimate (normal traffic, car / bike / walking; no public transport)
Our own command model understands the request; only the route itself comes from Google or OpenStreetMap.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import webbrowser

UA = "Jarvis-0 personal desktop assistant (single user)"
URL_MODE = {"driving": "driving", "two_wheeler": "driving", "walking": "walking", "transit": "transit",
            "bicycling": "bicycling"}
API_MODE = {"driving": "DRIVE", "two_wheeler": "TWO_WHEELER", "walking": "WALK", "transit": "TRANSIT",
            "bicycling": "BICYCLE"}
OSM_PROFILE = {"driving": "car", "two_wheeler": "car", "walking": "foot", "bicycling": "bike"}
_geo_cache: dict[str, tuple[float, float, str] | None] = {}
_last_nominatim = [0.0]


def maps_url(destination: str, origin: str | None, mode: str) -> str:
    q = {"api": "1", "destination": destination, "travelmode": URL_MODE.get(mode, "driving")}
    if origin:
        q["origin"] = origin
    return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(q)


def _http(url: str, data: dict | None = None, headers: dict | None = None, timeout: float = 8) -> dict:
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": UA, **(headers or {})},
                                 method="POST" if body else "GET")
    if body:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ------------------------------------------------------------------ Google Routes API (your key, live traffic)
def google_route(origin: str, destination: str, mode: str, key: str, http=_http) -> dict | None:
    body = {"origin": {"address": origin}, "destination": {"address": destination},
            "travelMode": API_MODE.get(mode, "DRIVE"), "computeAlternativeRoutes": mode != "transit",
            "languageCode": "en-IN", "units": "METRIC"}
    if mode in ("driving", "two_wheeler"):
        body["routingPreference"] = "TRAFFIC_AWARE"
    res = http("https://routes.googleapis.com/directions/v2:computeRoutes", body,
               {"X-Goog-Api-Key": key,
                "X-Goog-FieldMask": "routes.duration,routes.staticDuration,routes.distanceMeters,routes.description"})
    routes = res.get("routes") or []
    if not routes:
        return None
    best = min(routes, key=lambda r: int(str(r.get("duration", "0s")).rstrip("s") or 0))
    dur = int(str(best.get("duration", "0s")).rstrip("s") or 0)
    usual = int(str(best.get("staticDuration", "0s")).rstrip("s") or 0)
    return {"seconds": dur, "usual_seconds": usual or None, "meters": best.get("distanceMeters"),
            "via": best.get("description") or "", "source": "Google Maps", "traffic": mode in ("driving", "two_wheeler"),
            "alternatives": len(routes)}


# ------------------------------------------------------------------ OpenStreetMap (free, normal traffic)
def geocode(place: str, near: tuple[float, float] | None = None, http=_http) -> tuple[float, float, str] | None:
    key = f"{place}|{near}"
    if key in _geo_cache:
        return _geo_cache[key]
    wait = 1.1 - (time.time() - _last_nominatim[0])           # Nominatim's rule: at most one request a second
    if wait > 0:
        time.sleep(wait)
    q = {"q": place, "format": "json", "limit": "1"}
    if near:                                                  # prefer places near home
        lat, lon = near
        q.update(viewbox=f"{lon - 0.6},{lat + 0.6},{lon + 0.6},{lat - 0.6}", bounded="0")
    res = http("https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(q))
    _last_nominatim[0] = time.time()
    out = (float(res[0]["lat"]), float(res[0]["lon"]), res[0].get("display_name", place)) if res else None
    _geo_cache[key] = out
    return out


def osm_route(origin: str, destination: str, mode: str, near=None, http=_http) -> dict | None:
    if mode not in OSM_PROFILE:
        return None                                           # no public transport on OpenStreetMap routing
    a = geocode(origin, near, http)
    b = geocode(destination, near or (a[:2] if a else None), http)
    if not (a and b):
        return None
    prof = OSM_PROFILE[mode]
    url = (f"https://routing.openstreetmap.de/routed-{prof}/route/v1/driving/"
           f"{a[1]},{a[0]};{b[1]},{b[0]}?overview=false&alternatives=true")
    res = http(url)
    routes = res.get("routes") or []
    if not routes:
        return None
    best = min(routes, key=lambda r: r["duration"])
    return {"seconds": int(best["duration"]), "usual_seconds": None, "meters": int(best["distance"]), "via": "",
            "source": "OpenStreetMap", "traffic": False, "alternatives": len(routes)}


# ------------------------------------------------------------------ the action
def resolve_place(name: str | None, places: dict[str, str]) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    key = {"my home": "home", "ghar": "home", "my house": "home", "my office": "work", "office": "work",
           "my work": "work", "workplace": "work"}.get(key, key)
    return places.get(key) or name


def directions(args: dict, ctx, dry_run: bool, http=_http, open_url=webbrowser.open):
    from .base import ActionResult
    places = {k.lower(): v for k, v in (getattr(ctx.config, "places", None) or {}).items() if v}
    dest_said = str(args["destination"])
    origin_said = str(args["origin"]) if "origin" in args else None
    mode = str(args.get("mode", "driving"))
    dest = resolve_place(dest_said, places)
    origin = resolve_place(origin_said, places) if origin_said else places.get("home")
    data = {"destination": dest_said, "origin": origin_said or ("home" if origin else None), "mode": mode,
            "url": maps_url(dest, resolve_place(origin_said, places) if origin_said else None, mode)}
    if dest_said.lower() in ("home", "work", "office", "ghar", "my home", "my office") and dest == dest_said:
        data["missing_place"] = dest_said.lower()
    if dry_run:
        return ActionResult(f"Would open directions to {dest_said}", kind="directions", data=data)
    open_url(data["url"])
    route = None
    if origin and not data.get("missing_place"):
        from ..crypto import unprotect
        key = unprotect(getattr(ctx.config, "maps_api_key", "") or "")
        try:
            if key:
                route = google_route(origin, dest, mode, key, http)
            elif getattr(ctx.config, "maps_osm", True):
                route = osm_route(origin, dest, mode, http=http)
        except Exception as e:                                 # offline, quota, unknown place: Maps is open anyway
            data["route_error"] = str(e)[:120]
    if route:
        data.update(route)
    elif not origin:
        data["no_origin"] = True
    return ActionResult(f"Directions to {dest_said}", kind="directions", data=data)
