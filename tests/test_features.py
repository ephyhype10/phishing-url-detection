"""
Unit tests for feature_extractor.py.
Each test checks ONE specific piece of extraction logic in isolation.
Run with: pytest tests/test_features.py -v
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_extractor import extract_features, has_ip_address, shannon_entropy


# ---------- IP address detection ----------

def test_ip_detection_with_http():
    f = extract_features("http://192.168.1.1/login")
    assert f["has_ip"] == 1


def test_ip_detection_without_scheme():
    f = extract_features("192.168.1.1/login")
    assert f["has_ip"] == 1


def test_no_ip_for_normal_domain():
    f = extract_features("https://www.google.com")
    assert f["has_ip"] == 0


# ---------- @ symbol detection ----------

def test_at_symbol_detected():
    f = extract_features("http://safe.com@evil.com")
    assert f["has_at_symbol"] == 1


def test_no_at_symbol():
    f = extract_features("https://example.com")
    assert f["has_at_symbol"] == 0


# ---------- HTTPS flag ----------

def test_https_flag_true():
    f = extract_features("https://example.com")
    assert f["https_used"] == 1


def test_https_flag_false_for_http():
    f = extract_features("http://example.com")
    assert f["https_used"] == 0


# ---------- Suspicious keywords ----------

def test_suspicious_keyword_detected():
    f = extract_features("http://paypal-verify-login.com")
    assert f["has_suspicious_word"] == 1


def test_no_suspicious_keyword():
    f = extract_features("https://www.wikipedia.org")
    assert f["has_suspicious_word"] == 0


# ---------- Structural counts ----------

def test_dot_count():
    f = extract_features("http://a.b.c.example.com")
    assert f["num_dots"] == 4


def test_hyphen_count():
    f = extract_features("http://secure-login-update.com")
    assert f["num_hyphens"] == 2


def test_digit_count():
    f = extract_features("http://192.168.0.1")
    assert f["num_digits"] > 0


def test_subdomain_count():
    f = extract_features("http://a.b.c.example.com")
    assert f["num_subdomains"] >= 2


# ---------- Entropy ----------

def test_entropy_is_zero_for_empty_string():
    assert shannon_entropy("") == 0.0


def test_entropy_increases_with_randomness():
    low = shannon_entropy("aaaaaaaa")
    high = shannon_entropy("x7Qp9zL2")
    assert high > low


# ---------- Edge cases ----------

def test_empty_url_does_not_crash():
    f = extract_features("")
    assert f["url_length"] == 0
    assert f["has_ip"] == 0
    assert f["has_at_symbol"] == 0


def test_none_url_does_not_crash():
    f = extract_features(None)
    assert f["url_length"] == 0


def test_url_with_no_scheme_does_not_crash():
    f = extract_features("example.com/login")
    assert isinstance(f, dict)
    assert f["url_length"] > 0


def test_very_long_url_does_not_crash():
    long_url = "http://example.com/" + "a" * 3000
    f = extract_features(long_url)
    assert f["url_length"] > 3000


def test_whitespace_only_url():
    f = extract_features("   ")
    assert f["url_length"] == 0


def test_unicode_url_does_not_crash():
    f = extract_features("http://xn--pple-43d.com/login")
    assert isinstance(f, dict)


# ---------- Return type / schema consistency ----------

def test_returns_all_expected_keys():
    expected_keys = {
        "url_length", "num_dots", "num_hyphens", "num_underscores",
        "num_slashes", "num_digits", "num_special_chars", "has_at_symbol",
        "has_ip", "https_used", "num_subdomains", "has_suspicious_word",
        "url_entropy"
    }
    f = extract_features("https://example.com")
    assert set(f.keys()) == expected_keys


def test_all_values_are_numeric():
    f = extract_features("http://test-login.com/path?x=1")
    for key, value in f.items():
        assert isinstance(value, (int, float)), f"{key} is not numeric"