"""
Feature extraction module.
Converts a raw URL string into a dictionary of numeric features
that can be fed into any scikit-learn-compatible model.
"""

import re
import math
from urllib.parse import urlparse

SUSPICIOUS_WORDS = ["login", "verify", "secure", "update", "bank",
                     "confirm", "account", "signin", "ebay", "paypal"]


def shannon_entropy(s: str) -> float:
    """Measures randomness of characters in a string (higher = more random-looking)."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def has_ip_address(url: str) -> int:
    """Detects if the host is a raw IP address instead of a domain name."""
    pattern = r"^(https?:\/\/)?(\d{1,3}\.){3}\d{1,3}"
    return int(bool(re.match(pattern, url)))


def extract_features(url: str) -> dict:
    """
    Extracts lexical features from a URL.
    Returns a dictionary; keys become the column names used for training.
    Handles empty strings and malformed URLs without raising exceptions.
    """
    if url is None:
        url = ""
    url = str(url).strip()

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc or ""
    except Exception:
        host = ""

    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(not c.isalnum() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip": has_ip_address(url),
        "https_used": int(url.lower().startswith("https://")),
        "num_subdomains": max(host.count(".") - 1, 0) if host else 0,
        "has_suspicious_word": int(any(w in url.lower() for w in SUSPICIOUS_WORDS)),
        "url_entropy": round(shannon_entropy(url), 3),
    }
    return features


if __name__ == "__main__":
    # quick manual check when running this file directly
    test_urls = [
        "https://www.google.com",
        "http://192.168.1.1/login",
        "http://paypal-verify-account.com@evil.com",
        "",
    ]
    for u in test_urls:
        print(u, "->", extract_features(u))