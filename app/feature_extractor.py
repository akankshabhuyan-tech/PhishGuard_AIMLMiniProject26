
import re
import ipaddress
from urllib.parse import urlsplit

def extract_url_features(url):

    url = str(url).strip()

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url

    parsed = urlsplit(url)
    hostname = parsed.hostname or ""

    try:
        ipaddress.ip_address(hostname)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    labels = hostname.lower().split(".")
    tld = labels[-1] if len(labels) > 1 else ""

    no_of_subdomain = max(0, len(labels) - 2)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)

    encoded_chars = re.findall(
        r"%[0-9a-fA-F]{2}", url
    )

    equals_count = url.count("=")
    question_count = url.count("?")
    ampersand_count = url.count("&")

    other_special = sum(
        not c.isalnum() and c not in "=?&"
        for c in url
    )

    letter_ratio = letters / len(url) if url else 0
    digit_ratio = digits / len(url) if url else 0
    special_ratio = other_special / len(url) if url else 0
    obfuscation_ratio = len(encoded_chars) / len(url) if url else 0

    return {
        "URLLength": len(url),
        "DomainLength": len(hostname),
        "IsDomainIP": is_domain_ip,
        "TLDLength": len(tld),
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": int(len(encoded_chars) > 0),
        "NoOfObfuscatedChar": len(encoded_chars),
        "ObfuscationRatio": obfuscation_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letter_ratio,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digit_ratio,
        "NoOfEqualsInURL": equals_count,
        "NoOfQMarkInURL": question_count,
        "NoOfAmpersandInURL": ampersand_count,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": special_ratio,
        "IsHTTPS": int(parsed.scheme.lower() == "https")
    }
